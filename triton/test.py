import triton
import triton.language as tl


# 定义一个 Triton 内核，什么都不做，只是打印 "Hello, World!"
@triton.jit
def hello_world_kernel():
    print("Hello, World!")


# 主函数
def main():
    # 调用 Triton 内核
    hello_world_kernel[1]()  # [1] 表示只启动一个线程块，线程数为 1


if __name__ == "__main__":
    main()
