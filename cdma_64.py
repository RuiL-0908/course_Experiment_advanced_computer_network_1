# 用于解析命令行参数，使基础实验与选做噪声实验可以分别运行。
# 执行 python cdma_64.py 只运行基础实验；
# 执行 python cdma_64.py --with-noise 才运行选做实验。

import argparse

# 用于创建结果目录并保存 BER 数据和曲线。
from pathlib import Path

import numpy as np

NUM_USERS = 64
CODE_LENGTH = 64


def generate_walsh_codes(n=64):
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError("n must be a positive power of two")
    H = np.array([[1]], dtype=np.int64)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def ascii_to_bits(text):
    raw = text.encode("ascii")
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8))


def bits_to_ascii(bits):
    bits = np.asarray(bits, dtype=np.uint8)
    if len(bits) % 8 != 0:
        raise ValueError("bits length must be multiple of 8")
    return np.packbits(bits).tobytes().decode("ascii")


def spread_bits(bits, code):
    bipolar = 2 * np.asarray(bits, dtype=np.int64) - 1
    return bipolar[:, None] * code[None, :]


def combine_signals(spread_signals):
    return np.sum(spread_signals, axis=0)


def decode_user(combined_signal, code):
    correlation = combined_signal @ code / len(code)
    bits = (correlation > 0).astype(np.uint8)
    return bits, correlation


def main():
    H = generate_walsh_codes(CODE_LENGTH)
    print("Walsh matrix shape:", H.shape)
    print("First 8 user codes:\n", H[:8])

    # 验证 Walsh 码的正交性。
    # 理论上 H @ H.T 应严格等于 64 阶单位矩阵乘以码长 64，
    # 即同一组码的内积为 64，不同组码的内积为 0。
    expected = CODE_LENGTH * np.eye(NUM_USERS, dtype=np.int64) # 生成一个方阵，主对角线为 1，其余为 0
    gram_matrix = H @ H.T # 计算 Gram 矩阵，即码矩阵与其转置的乘积
    orthogonality_verified = np.array_equal(gram_matrix, expected)

    # 计算实际 Gram 矩阵和理论矩阵之间的最大误差。
    # 结果为 0 表示整个 64×64 矩阵均完全满足正交关系。
    maximum_orthogonality_error = np.max(np.abs(gram_matrix - expected))

    # 将验证结果直接打印。
    print("Orthogonality verified:", orthogonality_verified)
    print("Maximum orthogonality error:", maximum_orthogonality_error)

    # 如果正交性验证失败，立即终止程序，避免继续使用错误码矩阵。
    assert orthogonality_verified

    messages = [chr(ord("A") + i % 26) for i in range(NUM_USERS)]
    spread_signals = [
        spread_bits(ascii_to_bits(messages[i]), H[i])
        for i in range(NUM_USERS)
    ]
    combined_signal = combine_signals(spread_signals)
    print("Combined signal shape:", combined_signal.shape)
    print("First bit composite chips:\n", combined_signal[0])

    correct = 0
    recovered = []
    for i in range(NUM_USERS):
        bits, correlation = decode_user(combined_signal, H[i])
        text = bits_to_ascii(bits)
        recovered.append(text)
        is_correct = text == messages[i]
        correct += int(is_correct)
        print(
            f"User {i + 1:02d}: TX={messages[i]} "
            f"RX={text} correct={is_correct}"
        )

    print(f"Recovered correctly: {correct}/{NUM_USERS}")
    print(f"Accuracy: {correct / NUM_USERS:.2%}")
    assert recovered == messages


def run_noise_ber_experiment(
    num_chars=1000,
    noise_stds=(0, 2, 4, 8, 16),
    seed=42,
    output_dir="results",
):
    """运行选做的 AWGN 实验，并保存 BER 数据和曲线。"""

    # 创建结果目录，用于存放 CSV 数据和 BER 曲线。
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 为每个用户构造 num_chars 个字符的长消息。
    # 默认每用户发送 1000 个字符，以获得较稳定的 BER 统计结果。
    H = generate_walsh_codes(CODE_LENGTH)
    messages = [
        chr(ord("A") + user_index % 26) * num_chars
        for user_index in range(NUM_USERS)
    ]

    # 将 64 个用户的消息统一转换为比特矩阵。
    # transmitted_bits 的形状为 (用户数, 每用户比特数)。
    transmitted_bits = np.stack(
        [ascii_to_bits(message) for message in messages]
    )

    # 进行 0→-1、1→+1 的 BPSK 双极性映射。
    bipolar_bits = 2 * transmitted_bits.astype(np.int64) - 1

    # 用矩阵乘法一次完成 64 个用户的扩频与信号叠加。
    # (users, bits).T @ (users, chips) -> (bits, chips)
    combined_signal = bipolar_bits.T @ H

    # 固定随机种子，使每次实验都能得到可复现的噪声样本。
    rng = np.random.default_rng(seed)
    ber_values = []

    print("\nAWGN BER experiment")
    print(f"Characters per user: {num_chars}")
    print(f"Random seed: {seed}")
    print(f"Total transmitted bits: {transmitted_bits.size}")
    for noise_std in noise_stds:
        # 生成均值为 0、标准差为 noise_std 的高斯白噪声，
        # 并将其加入 64 用户叠加后的复合信号。
        noise = rng.normal(0, noise_std, combined_signal.shape)
        noisy_signal = combined_signal + noise

        # 同时使用全部 64 组 Walsh 码进行相关解扩。
        # 相关值大于 0 判为比特 1，否则判为比特 0。
        correlations = noisy_signal @ H.T / CODE_LENGTH
        recovered_bits = (correlations.T > 0).astype(np.uint8)

        # 比较发送比特与恢复比特，计算错误比特数和 BER。
        bit_errors = np.count_nonzero(recovered_bits != transmitted_bits)
        ber = bit_errors / transmitted_bits.size
        ber_values.append(ber)
        print(
            f"noise_std={noise_std:>2}: "
            f"bit_errors={bit_errors:>6}, BER={ber:.8f}"
        )

    # 将“噪声标准差—BER”结果保存为 CSV。
    result_table = np.column_stack((noise_stds, ber_values))
    csv_path = output_path / "ber_results.csv"
    np.savetxt(
        csv_path,
        result_table,
        delimiter=",",
        header="noise_std,ber",
        comments="",
        fmt=["%.1f", "%.10f"],
    )

    # Matplotlib 只在运行选做实验时导入，基础实验仍只依赖 NumPy。
    import matplotlib.pyplot as plt
    # 绘制 BER 随噪声标准差变化的曲线，并保存为 PNG 图片。
    figure_path = output_path / "ber_curve.png"
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(noise_stds, ber_values, marker="o", linewidth=2)
    plt.xlabel("Noise standard deviation (sigma)")
    plt.ylabel("Bit error rate (BER)")
    plt.title("64-user CDMA BER under AWGN")
    plt.xticks(noise_stds)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()

    print(f"BER data saved to: {csv_path}")
    print(f"BER curve saved to: {figure_path}")
    return result_table


if __name__ == "__main__":
    # 直接运行脚本只执行基础实验；加入 --with-noise 才执行选做实验。
    parser = argparse.ArgumentParser(
        description="64-user Walsh-code CDMA simulation"
    )
    parser.add_argument(
        "--with-noise",
        action="store_true",
        help="run the optional AWGN BER experiment",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "results"),
        help="directory for optional BER results",
    )
    args = parser.parse_args()

    # 始终执行指导书要求的 64 用户无噪声基础实验。
    main()

    # 根据命令行开关决定是否继续运行 AWGN–BER 选做实验。
    if args.with_noise:
        run_noise_ber_experiment(output_dir=args.output_dir)
