"""
测试进度跟踪系统
"""

import sys
import os
import asyncio
import time

# 添加项目路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

from progress_tracker import (
    create_progress_tracker,
    get_progress_tracker,
    remove_progress_tracker,
    ProgressEvent,
    ProgressEventType,
)


async def test_progress_tracker():
    """测试进度跟踪器的基本功能"""
    print("=" * 60)
    print("测试进度跟踪系统")
    print("=" * 60)

    session_id = "test-session-123"

    # 1. 创建进度跟踪器
    print("\n1. 创建进度跟踪器")
    tracker = create_progress_tracker(session_id)
    print(f"   ✓ 创建成功: session_id={session_id}")

    # 2. 创建进度回调
    print("\n2. 创建进度回调")
    callback = tracker.create_callback()
    print("   ✓ 回调创建成功")

    # 3. 发送测试进度事件
    print("\n3. 发送测试进度事件")

    # 模拟节点开始
    callback(
        event_type="node_start",
        message="Starting supervisor node",
        node_name="supervisor",
        progress_percent=0,
    )
    print("   ✓ 发送: node_start - supervisor")

    await asyncio.sleep(0.2)

    # 模拟节点进度
    callback(
        event_type="node_progress",
        message="Processing task classification",
        node_name="supervisor",
        progress_percent=50,
    )
    print("   ✓ 发送: node_progress - supervisor (50%)")

    await asyncio.sleep(0.2)

    # 模拟节点完成
    callback(
        event_type="node_complete",
        message="Task classified as GENERAL_QA",
        node_name="supervisor",
        progress_percent=100,
    )
    print("   ✓ 发送: node_complete - supervisor (100%)")

    # 4. 接收进度事件
    print("\n4. 接收进度事件")
    events = []

    while True:
        event = await tracker.get_event(timeout=0.1)
        if event is None:
            break
        events.append(event)
        print(f"   ✓ 接收: {event.event_type} - {event.message}")

    print(f"\n   总共接收 {len(events)} 个事件")

    # 5. 验证事件内容
    print("\n5. 验证事件内容")
    assert len(events) == 3, f"Expected 3 events, got {len(events)}"
    assert events[0].event_type == ProgressEventType.NODE_START
    assert events[1].event_type == ProgressEventType.NODE_PROGRESS
    assert events[2].event_type == ProgressEventType.NODE_COMPLETE
    print("   ✓ 所有事件验证通过")

    # 6. 清理
    print("\n6. 清理进度跟踪器")
    remove_progress_tracker(session_id)
    print("   ✓ 清理完成")

    # 7. 验证清理
    print("\n7. 验证清理")
    tracker_after = get_progress_tracker(session_id)
    assert tracker_after is None, "Tracker should be removed"
    print("   ✓ 验证通过: tracker已移除")

    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)


async def test_concurrent_events():
    """测试并发事件处理"""
    print("\n" + "=" * 60)
    print("测试并发事件处理")
    print("=" * 60)

    session_id = "test-concurrent-456"
    tracker = create_progress_tracker(session_id)
    callback = tracker.create_callback()

    # 并发发送多个事件
    print("\n1. 并发发送10个事件")
    for i in range(10):
        callback(event_type="info", message=f"Event {i + 1}", details={"index": i})

    print("   ✓ 事件发送完成")

    # 接收所有事件
    print("\n2. 接收所有事件")
    events = []
    while True:
        event = await tracker.get_event(timeout=0.1)
        if event is None:
            break
        events.append(event)

    print(f"   ✓ 接收到 {len(events)} 个事件")
    assert len(events) == 10, f"Expected 10 events, got {len(events)}"

    remove_progress_tracker(session_id)
    print("\n✅ 并发测试通过！")


if __name__ == "__main__":
    # 运行基本测试
    asyncio.run(test_progress_tracker())

    # 运行并发测试
    asyncio.run(test_concurrent_events())

    print("\n" + "🎉 " * 20)
    print("所有测试完成！进度跟踪系统工作正常。")
    print("🎉 " * 20)
