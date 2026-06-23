import requests

# 1. 测试 Python 自身是否连通 LM Studio（可选）
print("🔍 测试 Python → LM Studio 连通性...")
try:
    from openai import OpenAI
    client = OpenAI(base_url="http://127.0.0.1:11435/v1", api_key="x")
    # 这里只简单列出模型，验证连接
    models = client.models.list()
    print(f"✅ LM Studio 连接成功，已加载模型：{models.data[0].id}")
except Exception as e:
    print(f"❌ LM Studio 连接失败：{e}")

# 2. 测试 Node 上传接口
print("\n📤 测试上传文件到 Node 网关...")
url = "http://localhost:3000/api/upload"
file_path = "C:/Users/Administrator/Desktop/test.txt"

try:
    with open(file_path, "rb") as f:
        files = {"file": f}
        resp = requests.post(url, files=files)
    data = resp.json()
    if resp.status_code == 200 and "summary" in data:
        print(f"✅ 上传成功！")
        print(f"   文件名：{data.get('fileName')}")
        print(f"   摘要：{data.get('summary')}")
    else:
        print(f"⚠️ 返回异常：{data}")
except FileNotFoundError:
    print(f"❌ 文件不存在：{file_path}，请在桌面创建 test.txt")
except requests.ConnectionError:
    print("❌ 无法连接 Node 网关，确认 Node 是否在 3000 端口运行")
except Exception as e:
    print(f"❌ 请求失败：{e}")