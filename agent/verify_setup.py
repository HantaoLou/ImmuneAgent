"""
验证环境配置是否正确设置
运行此脚本检查所有必需的依赖和配置
"""

import sys
from pathlib import Path

def check_python_version():
    """检查 Python 版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 12):
        print(f"❌ Python 版本过低: {sys.version}")
        print("   需要 Python >= 3.12")
        return False
    print(f"✅ Python 版本: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """检查关键依赖"""
    dependencies = [
        ("langchain", "LangChain"),
        ("langgraph", "LangGraph"),
        ("langchain_mcp_adapters", "LangChain MCP Adapters"),
        ("langchain_openai", "LangChain OpenAI"),
        ("pydantic", "Pydantic"),
    ]
    
    missing = []
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name} 已安装")
        except ImportError:
            print(f"❌ {name} 未安装")
            missing.append(name)
    
    return len(missing) == 0

def check_api_keys():
    """检查 API Keys 配置"""
    import os
    from config.api_keys import APIKeys
    
    keys_to_check = [
        ("OPENAI_API_KEY", APIKeys.OPENAI_API_KEY),
    ]
    
    all_ok = True
    for key_name, key_value in keys_to_check:
        env_value = os.getenv(key_name)
        if env_value and env_value != "your openai api key":
            print(f"✅ {key_name} 已从环境变量设置")
        elif key_value and key_value != "your openai api key":
            print(f"⚠️  {key_name} 使用配置文件中的默认值（建议使用环境变量）")
        else:
            print(f"❌ {key_name} 未配置")
            all_ok = False
    
    return all_ok

def check_project_structure():
    """检查项目结构"""
    required_files = [
        "config/api_keys.py",
        "config/config.py",
        "usecases/immunity/start_improved_workflow.py",
    ]
    
    all_ok = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} 存在")
        else:
            print(f"❌ {file_path} 不存在")
            all_ok = False
    
    return all_ok

def main():
    """主函数"""
    print("=" * 70)
    print("🔍 环境配置验证")
    print("=" * 70)
    print()
    
    checks = [
        ("Python 版本", check_python_version),
        ("项目结构", check_project_structure),
        ("依赖包", check_dependencies),
        ("API Keys", check_api_keys),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n📋 检查: {name}")
        print("-" * 70)
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ 检查失败: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 70)
    print("📊 验证结果摘要")
    print("=" * 70)
    
    all_passed = True
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")
        if not result:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 所有检查通过！环境配置正确。")
        print("\n您可以运行:")
        print("  python usecases/immunity/start_improved_workflow.py --query \"您的查询\"")
    else:
        print("⚠️  部分检查未通过，请根据上述提示修复问题。")
        print("\n详细配置指南请查看: SETUP_GUIDE.md")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
