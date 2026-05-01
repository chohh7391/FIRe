import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Compare Robot and Sim Observation Data")
    parser.add_argument("--robot", type=str, required=True, help="Path to robot csv file")
    parser.add_argument("--sim", type=str, required=True, help="Path to sim csv file")
    args = parser.parse_args()

    # 데이터 로드
    df_robot = pd.read_csv(args.robot)
    df_sim = pd.read_csv(args.sim)

    # 로봇 데이터와 시뮬레이션 데이터(obs_*)의 매핑 정의
    # (주의: Sim의 obs 인덱스가 아래 순서대로 되어있다고 가정합니다)
    groups = {
        "Position (pos)": {
            "robot": ["fingertip_pos_rel_fixed_1", "fingertip_pos_rel_fixed_2", "fingertip_pos_rel_fixed_3"],
            "sim": ["obs_0", "obs_1", "obs_2"]
        },
        "Quaternion (quat)": {
            "robot": ["fingertip_quat_0", "fingertip_quat_1", "fingertip_quat_2", "fingertip_quat_3"],
            "sim": ["obs_3", "obs_4", "obs_5", "obs_6"]
        },
        "Linear Velocity (linvel)": {
            "robot": ["ee_linvel_0", "ee_linvel_1", "ee_linvel_2"],
            "sim": ["obs_7", "obs_8", "obs_9"]
        },
        "Angular Velocity (angvel)": {
            "robot": ["ee_angvel_0", "ee_angvel_1", "ee_angvel_2"],
            "sim": ["obs_10", "obs_11", "obs_12"]
        },
        "FT Force (ft_force)": {
            # force_threshold 가 포함될 수도 있으나 여기서는 힘 성분(x, y, z)을 우선 매핑
            "robot": ["ft_force_0", "ft_force_1", "ft_force_2"],
            "sim": ["obs_14", "obs_15", "obs_16"]
        }
    }

    # 각 그룹(항목)별로 새로운 창(Figure)을 생성하여 플롯
    for group_name, cols in groups.items():
        robot_cols = cols["robot"]
        sim_cols = cols["sim"]
        
        num_plots = len(robot_cols)
        fig, axes = plt.subplots(num_plots, 1, figsize=(10, 3 * num_plots))
        fig.suptitle(f"{group_name} Comparison", fontsize=16)

        # 서브플롯이 1개일 경우 배열로 만들어줌
        if num_plots == 1:
            axes = [axes]

        for i in range(num_plots):
            ax = axes[i]
            
            # 데이터 길이가 다를 수 있으므로 각각의 인덱스(Step)를 기준으로 플롯
            if robot_cols[i] in df_robot.columns:
                ax.plot(df_robot.index, df_robot[robot_cols[i]], label=f"Robot: {robot_cols[i]}", alpha=0.8, linewidth=2)
            else:
                print(f"Warning: Column {robot_cols[i]} not found in robot csv.")

            if sim_cols[i] in df_sim.columns:
                ax.plot(df_sim.index, df_sim[sim_cols[i]], label=f"Sim: {sim_cols[i]}", alpha=0.8, linestyle='--')
            else:
                print(f"Warning: Column {sim_cols[i]} not found in sim csv.")

            ax.set_xlabel("Steps")
            ax.set_ylabel("Value")
            ax.legend(loc='upper right')
            ax.grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96]) # suptitle과 겹치지 않게 여백 조정

    # 모든 창을 한 번에 띄웁니다.
    plt.show()

if __name__ == "__main__":
    main()