import os
import subprocess
import sys

def parse_slice_argument(slice_arg):
    """
    解析切片参数，支持单个整数或元组形式。
    
    参数：
    slice_arg (str): 表示切片的字符串，如'10'或'(10,20)'。
    
    返回：
    tuple: 表示切片的元组。
    """
    if ',' in slice_arg:
        # 处理元组形式，如'(10,20)'
        start, end = slice_arg.strip('()').split(',')
        return (int(start), int(end))
    else:
        # 处理单个整数形式，如'10'
        index = int(slice_arg)
        return (index, index + 1)  # 返回切片，表示单个文件

def run_python_tests(test_directory='./tests', slice=(0, None)):
    """
    运行指定目录下的Python测试文件，根据提供的切片。
    
    参数：
    test_directory (str): 包含测试文件的目录路径。
    slice (tuple): 要运行文件的切片，形如(start, end)。
    """
    # 确保目录存在
    if not os.path.exists(test_directory):
        print("Test directory does not exist.")
        return

    # 获取所有Python文件
    files = [file for file in os.listdir(test_directory) if file.endswith('.py')]
    
    # 根据切片运行测试文件
    for file in files[slice[0]:slice[1]]:
        file_path = os.path.join(test_directory, file)
        print(f"Running {file_path}...")
        # 运行Python文件并捕获输出
        try:
            result = subprocess.run(['python', file_path], capture_output=True, text=True)
            print("Output:")
            print(result.stdout)
            print("Errors:")
            print(result.stderr)
        except Exception as e:
            print(f"Error running {file_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        slice_arg = sys.argv[1]
        slice_range = parse_slice_argument(slice_arg)
    else:
        slice_range = (0, 10)  # 默认值

    run_python_tests(slice=slice_range)
