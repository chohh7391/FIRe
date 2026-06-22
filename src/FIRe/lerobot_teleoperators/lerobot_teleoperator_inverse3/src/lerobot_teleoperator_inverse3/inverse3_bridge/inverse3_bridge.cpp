// Standalone server that wraps the Haplay SDK and exposes a simple
// line-based stdio protocol so Python can communicate via subprocess.
//
// Protocol (all lines are space-separated ASCII):
//
//   Python → Server          Server → Python
//   ─────────────────────    ──────────────────────────────────────────────
//   OPEN <inv3> <grip>   →  OK
//                        or  ERR <message>
//   GET_STATE            →  STATE <px> <py> <pz> <vx> <vy> <vz>
//                                <qw> <qx> <qy> <qz> <buttons> <battery>
//   SEND_FORCE <fx> <fy> <fz> → STATE ... (same format, updated pos/vel)
//   CLOSE                →  (server exits cleanly)
//
// Quaternion order: WXYZ (Haply SDK native format).

#include <cstdint>
#include <cstring>
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <memory>
#include <thread>

#include "Inverse3.h"
#include "Handle.h"
#include "SerialStream.h"

using namespace Haply::HardwareAPI;

// ── Device context ────────────────────────────────────────────────────────────

static std::unique_ptr<IO::SerialStream>   g_inv3_stream;
static std::unique_ptr<Devices::Inverse3>  g_inv3;
static std::unique_ptr<IO::SerialStream>   g_grip_stream;
static std::unique_ptr<Devices::Handle>    g_grip;

// Whether the VerseGrip handle has produced at least one valid frame. The
// wireless handle may be powered off when OPEN runs; in that case we still
// connect (Inverse3 translation works) and recover automatically once the
// handle wakes up and starts sending valid frames.
static bool g_grip_ok = false;

// Both devices stream at ~1 kHz. If the client polls slowly (e.g. a 15 Hz
// teleop loop), the OS serial buffers overflow between reads and frames desync
// (the grip's error_flag trips and buttons get dropped). So a background thread
// polls both devices continuously and caches the freshest state; GET_STATE just
// returns the cache, decoupling the client's rate from the device rate. This
// also keeps the Inverse3 fed at high rate, which it needs to stay stable.
struct CachedState {
    float pos[3]   = {0, 0, 0};
    float vel[3]   = {0, 0, 0};
    float quat[4]  = {1, 0, 0, 0};  // WXYZ, identity until a valid grip frame
    int   buttons  = 0;
    float battery  = 0.0f;
};
static std::mutex          g_state_mtx;
static CachedState         g_state;
static std::atomic<float>  g_force[3] = {{0.0f}, {0.0f}, {0.0f}};
static std::atomic<bool>   g_poll_running{false};
static std::thread         g_poll_thread;

// Discard any stale bytes sitting in the stream's input buffer so the next
// command/response pair starts cleanly synchronised.
static void drain_input(IO::SerialStream* s) {
    s->clear();
    for (int guard = 0; guard < 4096 && s->Available() > 0; ++guard)
        s->get();
    s->clear();
}

// The Inverse3 must receive DeviceWakeup before any other command. The very
// first attempt after opening the port often times out (the device may still
// be booting / the stream needs to synchronise), which leaves the iostream in
// a fail state and makes every later read return immediately with zeros.
// Retry with the error flags cleared until the device reports a valid id.
static bool wakeup_inverse3(int max_attempts = 20, int delay_ms = 300) {
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        drain_input(g_inv3_stream.get());
        auto info = g_inv3->DeviceWakeup();
        if (g_inv3_stream->good() && info.device_id != 0)
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }
    return false;
}

// Wireless VerseGrip: a valid status frame has error_flag == 0. When the
// handle is asleep/off the dongle still streams frames but with error_flag set
// (observed value 99). Best-effort: try briefly so OPEN doesn't block forever.
static bool wakeup_versegrip(int max_attempts = 4, int delay_ms = 100) {
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        g_grip_stream->clear();
        g_grip->SendDeviceWakeup();
        auto resp = g_grip->GetVersegripStatus(false);
        if (g_grip_stream->good() && resp.error_flag == 0)
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }
    return false;
}

static void open_devices(const std::string& inv3_port, const std::string& grip_port) {
    g_inv3_stream = std::make_unique<IO::SerialStream>(inv3_port.c_str(), true);
    if (!g_inv3_stream->good())
        throw std::runtime_error("Cannot open Inverse3 port: " + inv3_port);

    g_inv3 = std::make_unique<Devices::Inverse3>(g_inv3_stream.get());
    if (!wakeup_inverse3())
        throw std::runtime_error("Inverse3 DeviceWakeup failed (no response on " + inv3_port + ")");

    Devices::Inverse3::GravityCompensationPayload gc;
    gc.enabled = false;
    gc.gravity_scale_factor = 0.0f;
    g_inv3->SetGravityCompensation(gc);
    std::cerr << "[info] Inverse3 gravity compensation: disabled\n";

    // Make session startup explicit: motors stay active for state polling, but
    // no external haptic force or gravity compensation is applied.
    const float zero_force[3] = {0.f, 0.f, 0.f};
    g_inv3->SendEndEffectorForce(zero_force);
    g_force[0].store(0.0f);
    g_force[1].store(0.0f);
    g_force[2].store(0.0f);

    g_grip_stream = std::make_unique<IO::SerialStream>(grip_port.c_str(), true);
    if (!g_grip_stream->good())
        throw std::runtime_error("Cannot open VerseGrip port: " + grip_port);

    g_grip = std::make_unique<Devices::Handle>(g_grip_stream.get());
    g_grip_ok = wakeup_versegrip();
    if (!g_grip_ok)
        std::cerr << "[warn] VerseGrip handle not responding (only invalid frames). "
                     "Power on / pair the handle; orientation and buttons stay neutral "
                     "until it sends valid frames.\n";
}

// The Inverse3 firmware paired with SDK 0.2.8 streams unsolicited 5-byte
// heartbeat frames (0xBA 0x02 0x02 <2 bytes>) interleaved with the cursor-state
// response, which desyncs the SDK's own EndEffectorForce reader. So we send the
// force with the SDK (that half works — torque is applied) and parse the state
// response ourselves:
//   state frame  = 0x2B <pos.x pos.y pos.z vel.x vel.y vel.z> (6 LE float32)
//   heartbeat    = 0xBA 0x02 0x02 <2 bytes>   (skipped)
static constexpr uint8_t INV3_STATE_HEADER = 0x2B;
static constexpr uint8_t INV3_HEARTBEAT_HEADER = 0xBA;

static int read_byte_blocking(IO::SerialStream* s, std::chrono::steady_clock::time_point deadline) {
    while (std::chrono::steady_clock::now() < deadline) {
        s->clear();
        if (s->Available() > 0) {
            int c = s->get();
            if (c >= 0) return c & 0xFF;
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    return -1;
}

// Send the given force, then read back one cursor-state frame. Returns false on
// timeout (state left untouched). force == nullptr sends zero force.
static bool inv3_force_and_read(const float* force,
                                Devices::Inverse3::EndEffectorStateResponse& out,
                                int timeout_ms = 100) {
    static const float kZero[3] = {0.f, 0.f, 0.f};
    g_inv3_stream->clear();
    g_inv3->SendEndEffectorForce(force ? force : kZero);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (true) {
        int h = read_byte_blocking(g_inv3_stream.get(), deadline);
        if (h < 0) return false;
        if (h == INV3_HEARTBEAT_HEADER) {
            for (int i = 0; i < 4; ++i)
                if (read_byte_blocking(g_inv3_stream.get(), deadline) < 0) return false;
            continue;
        }
        if (h != INV3_STATE_HEADER) continue;  // skip stray byte, resync

        uint8_t buf[24];
        for (int i = 0; i < 24; ++i) {
            int b = read_byte_blocking(g_inv3_stream.get(), deadline);
            if (b < 0) return false;
            buf[i] = static_cast<uint8_t>(b);
        }
        std::memcpy(out.position, buf, 12);
        std::memcpy(out.velocity, buf + 12, 12);
        return true;
    }
}

// Background poll: continuously drive both devices and cache the freshest state.
static void poll_loop() {
    while (g_poll_running.load()) {
        float force[3] = {g_force[0].load(), g_force[1].load(), g_force[2].load()};
        Devices::Inverse3::EndEffectorStateResponse inv3{};
        bool inv3_ok = inv3_force_and_read(force, inv3, 20);

        auto grip = g_grip->GetVersegripStatus(false);
        bool grip_valid = (grip.error_flag == 0);
        if (grip_valid) g_grip_ok = true;

        std::lock_guard<std::mutex> lk(g_state_mtx);
        if (inv3_ok) {
            std::memcpy(g_state.pos, inv3.position, sizeof(g_state.pos));
            std::memcpy(g_state.vel, inv3.velocity, sizeof(g_state.vel));
        }
        // Only overwrite grip fields on a valid frame; transient invalid frames
        // (handle radio noise) keep the last good orientation/buttons instead of
        // flickering to identity / dropping a held button.
        if (grip_valid) {
            g_state.quat[0] = grip.quaternion[0];
            g_state.quat[1] = grip.quaternion[1];
            g_state.quat[2] = grip.quaternion[2];
            g_state.quat[3] = grip.quaternion[3];
            g_state.buttons = static_cast<int>(grip.buttons);
            g_state.battery = grip.battery_level;
        }
    }
}

static void emit_cached_state() {
    CachedState s;
    {
        std::lock_guard<std::mutex> lk(g_state_mtx);
        s = g_state;
    }
    std::cout << "STATE"
              << " " << s.pos[0] << " " << s.pos[1] << " " << s.pos[2]
              << " " << s.vel[0] << " " << s.vel[1] << " " << s.vel[2]
              << " " << s.quat[0] << " " << s.quat[1] << " " << s.quat[2] << " " << s.quat[3]
              << " " << s.buttons
              << " " << s.battery
              << "\n";
    std::cout.flush();
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    std::string line;

    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;

        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;

        if (cmd == "OPEN") {
            std::string inv3_port, grip_port;
            ss >> inv3_port >> grip_port;
            try {
                open_devices(inv3_port, grip_port);
                g_poll_running.store(true);
                g_poll_thread = std::thread(poll_loop);  // start high-rate polling
                std::cout << (g_grip_ok ? "OK\n" : "OK GRIP_UNAVAILABLE\n");
                std::cout.flush();
            } catch (const std::exception& e) {
                std::cout << "ERR " << e.what() << "\n";
                std::cout.flush();
            }

        } else if (cmd == "GET_STATE") {
            emit_cached_state();

        } else if (cmd == "SEND_FORCE") {
            float force[3];
            ss >> force[0] >> force[1] >> force[2];
            g_force[0].store(force[0]);
            g_force[1].store(force[1]);
            g_force[2].store(force[2]);
            emit_cached_state();

        } else if (cmd == "CLOSE") {
            // Stop the poll thread, then zero the force. We intentionally do NOT
            // disable gravity compensation here: SetGravityCompensation persisted
            // as "disabled" on the device and left it limp, and open_devices
            // re-enables it each session anyway.
            g_poll_running.store(false);
            if (g_poll_thread.joinable())
                g_poll_thread.join();
            if (g_inv3) {
                const float zero[3] = {0.f, 0.f, 0.f};
                g_inv3->SendEndEffectorForce(zero);
            }
            break;

        } else {
            std::cout << "ERR Unknown command: " << cmd << "\n";
            std::cout.flush();
        }
    }

    return 0;
}
