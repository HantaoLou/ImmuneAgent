#!/usr/bin/env python3
"""
Test script for CodeAct Agent
Tests the code generation and execution functionality
"""

import sys
import os

# Ensure the package directory is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from agent import CodeActAgent


def test_simple_calculation():
    """Test simple calculation task."""
    print("=" * 60)
    print("Test 1: Simple Calculation")
    print("=" * 60)
    
    agent = CodeActAgent(
        llm='deepseek-chat',
        source='Custom',
        base_url='https://api.deepseek.com',
        api_key='sk-f8dda65f37b946b3ba12963a1dbdd0ed',
        timeout_seconds=300,
    )
    
    task = "Calculate the sum of numbers from 1 to 100 and print the result."
    
    print(f"Task: {task}")
    print("-" * 60)
    
    log, final_output = agent.go(task)
    
    print("\n" + "=" * 60)
    print("Execution Log:")
    print("=" * 60)
    for entry in log:
        print(entry)
        print("-" * 40)
    
    print("\n" + "=" * 60)
    print("Final Output:")
    print("=" * 60)
    print(final_output)
    
    return True


def test_data_analysis():
    """Test data analysis task."""
    print("\n" + "=" * 60)
    print("Test 2: Data Analysis")
    print("=" * 60)
    
    agent = CodeActAgent(
        llm='deepseek-chat',
        source='Custom',
        base_url='https://api.deepseek.com',
        api_key='sk-f8dda65f37b946b3ba12963a1dbdd0ed',
        timeout_seconds=300,
    )
    
    task = """
    Create a simple dataset with 10 random numbers between 1 and 100,
    then calculate and print:
    1. The mean
    2. The standard deviation
    3. The min and max values
    """
    
    print(f"Task: {task}")
    print("-" * 60)
    
    log, final_output = agent.go(task)
    
    print("\n" + "=" * 60)
    print("Execution Log:")
    print("=" * 60)
    for entry in log:
        print(entry)
        print("-" * 40)
    
    print("\n" + "=" * 60)
    print("Final Output:")
    print("=" * 60)
    print(final_output)
    
    return True


def test_file_operations():
    """Test file operations task."""
    print("\n" + "=" * 60)
    print("Test 3: File Operations")
    print("=" * 60)
    
    agent = CodeActAgent(
        llm='deepseek-chat',
        source='Custom',
        base_url='https://api.deepseek.com',
        api_key='sk-f8dda65f37b946b3ba12963a1dbdd0ed',
        timeout_seconds=300,
    )
    
    task = """
    1. Create a file called 'test_output.txt' in the current directory
    2. Write "Hello from CodeAct Agent!" to the file
    3. Read the file content and print it
    4. Delete the file
    """
    
    print(f"Task: {task}")
    print("-" * 60)
    
    log, final_output = agent.go(task)
    
    print("\n" + "=" * 60)
    print("Execution Log:")
    print("=" * 60)
    for entry in log:
        print(entry)
        print("-" * 40)
    
    print("\n" + "=" * 60)
    print("Final Output:")
    print("=" * 60)
    print(final_output)
    
    return True


def main():
    """Run all tests."""
    print("🚀 CodeAct Agent Test Suite")
    print("=" * 60)
    
    # Run tests
    try:
        test_simple_calculation()
        print("\n✅ Test 1 Passed!")
    except Exception as e:
        print(f"\n❌ Test 1 Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Uncomment to run additional tests
    # try:
    #     test_data_analysis()
    #     print("\n✅ Test 2 Passed!")
    # except Exception as e:
    #     print(f"\n❌ Test 2 Failed: {e}")
    
    # try:
    #     test_file_operations()
    #     print("\n✅ Test 3 Passed!")
    # except Exception as e:
    #     print(f"\n❌ Test 3 Failed: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 Test Suite Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
