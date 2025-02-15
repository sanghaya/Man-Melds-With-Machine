"""
Script for running both the camera tracking and machine controller scripts asynchronously
Call this script when running the app from a single machine - ie RUN_MODE = async
"""

import asyncio
import hand_tracking_v2, control_machine
import time

async def run_scripts():
    """Simultaneously call 2 scripts"""
    print("Starting main script...")
    
    try:
        # Create shared queue for async communication
        data_queue = asyncio.Queue()
        
        # Create and run tasks
        tracking_task = asyncio.create_task(hand_tracking_v2.main(data_queue))
        control_task = asyncio.create_task(control_machine.main(data_queue))
        
        # Wait for both tasks
        await asyncio.gather(tracking_task, control_task)
        
    except Exception as e:
        print(f"Error in main script: {e}")
    finally:
        print("Shutting down...")

if __name__ == "__main__":
    # Test mouse control at startup
    print("Testing mouse controller...")
    mouse = control_machine.MouseController()
    try:
        original_pos = mouse.position
        print(f"Original position: {original_pos}")
        test_pos = (100, 100)
        mouse.position = test_pos
        print(f"Moved to test position: {test_pos}")
        time.sleep(1)
        mouse.position = original_pos
        print("Mouse controller test successful")
    except Exception as e:
        print(f"Mouse controller test failed: {e}")
        
    # Run the main script
    asyncio.run(run_scripts())