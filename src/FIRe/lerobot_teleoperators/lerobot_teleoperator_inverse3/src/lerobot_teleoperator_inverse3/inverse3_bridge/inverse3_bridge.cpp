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
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <memory>

#include "Inverse3.h"
#include "Handle.h"
#include "SerialStream.h"

using namespace Haply::HardwareAPI;

// ── Device context ────────────────────────────────────────────────────────────

static std::unique_ptr<IO::SerialStream>   g_inv3_stream;
static std::unique_ptr<Devices::Inverse3>  g_inv3;
static std::unique_ptr<IO::SerialStream>   g_grip_stream;
static std::unique_ptr<Devices::Handle>    g_grip;

static void open_devices(const std::string& inv3_port, const std::string& grip_port) {
    g_inv3_stream = std::make_unique<IO::SerialStream>(inv3_port.c_str(), true);
    if (!g_inv3_stream->good())
        throw std::runtime_error("Cannot open Inverse3 port: " + inv3_port);

    g_inv3 = std::make_unique<Devices::Inverse3>(g_inv3_stream.get());
    g_inv3->DeviceWakeup();
    Devices::Inverse3::GravityCompensationPayload gc;
    gc.enabled = true;
    gc.gravity_scale_factor = 0.75f;
    g_inv3->SetGravityCompensation(gc);

    g_grip_stream = std::make_unique<IO::SerialStream>(grip_port.c_str(), true);
    if (!g_grip_stream->good())
        throw std::runtime_error("Cannot open VerseGrip port: " + grip_port);

    g_grip = std::make_unique<Devices::Handle>(g_grip_stream.get());
    g_grip->SendDeviceWakeup();
    g_grip->Receive();
}

static void print_state(const Devices::Inverse3::EndEffectorStateResponse& inv3,
                        const Devices::Handle::VersegripStatusResponse&    grip) {
    std::cout << "STATE"
              << " " << inv3.position[0]
              << " " << inv3.position[1]
              << " " << inv3.position[2]
              << " " << inv3.velocity[0]
              << " " << inv3.velocity[1]
              << " " << inv3.velocity[2]
              << " " << grip.quaternion[0]   // W
              << " " << grip.quaternion[1]   // X
              << " " << grip.quaternion[2]   // Y
              << " " << grip.quaternion[3]   // Z
              << " " << static_cast<int>(grip.buttons)
              << " " << grip.battery_level
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
                std::cout << "OK\n";
                std::cout.flush();
            } catch (const std::exception& e) {
                std::cout << "ERR " << e.what() << "\n";
                std::cout.flush();
            }

        } else if (cmd == "GET_STATE") {
            Devices::Inverse3::EndEffectorForceRequest zero{};
            auto inv3_resp = g_inv3->EndEffectorForce(zero, true);
            auto grip_resp = g_grip->GetVersegripStatus(false);
            print_state(inv3_resp, grip_resp);

        } else if (cmd == "SEND_FORCE") {
            float fx, fy, fz;
            ss >> fx >> fy >> fz;
            Devices::Inverse3::EndEffectorForceRequest req{};
            req.force[0] = fx;
            req.force[1] = fy;
            req.force[2] = fz;
            auto inv3_resp = g_inv3->EndEffectorForce(req, true);
            auto grip_resp = g_grip->GetVersegripStatus(false);
            print_state(inv3_resp, grip_resp);

        } else if (cmd == "CLOSE") {
            break;

        } else {
            std::cout << "ERR Unknown command: " << cmd << "\n";
            std::cout.flush();
        }
    }

    return 0;
}
