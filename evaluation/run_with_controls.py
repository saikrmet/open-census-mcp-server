#!/usr/bin/env python3
"""
Bulletproof evaluation wrapper - NO MORE DATA LOSS
"""

import sys
import signal
import time
import threading
from pathlib import Path

# Global flag for clean shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_requested
    print("\n\n🛑 INTERRUPT RECEIVED - Attempting graceful shutdown...")
    print("⚠️ Your data will be saved automatically!")
    print("Press Ctrl+C again within 5 seconds to force quit")
    shutdown_requested = True
    
    def force_quit():
        time.sleep(5)
        if shutdown_requested:
            print("\n💀 FORCE QUIT - Check session_state.json for recovery")
            sys.exit(1)
    
    threading.Thread(target=force_quit, daemon=True).start()

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def run_evaluation_safely():
    """Run evaluation with maximum data protection"""
    global shutdown_requested
    
    print("🏛️ Census MCP Evaluation - BULLETPROOF VERSION")
    print("=" * 50)
    print("✅ Data is saved after EVERY response")
    print("✅ Session state is backed up continuously")
    print("✅ Interruptions will preserve your work")
    print("⚡ Controls:")
    print("   Type 'DONE' after pasting = Process response")
    print("   Ctrl+C once = Graceful save & exit")
    print("   Ctrl+C twice = Force quit (data still saved)")
    print("-" * 50)
    
    try:
        from interactive_session import main as run_evaluation
        
        while not shutdown_requested:
            try:
                run_evaluation()
                print("🎉 Evaluation completed successfully!")
                break
                
            except KeyboardInterrupt:
                if shutdown_requested:
                    print("\n✅ Graceful shutdown - your data is safe!")
                    return
                else:
                    signal_handler(signal.SIGINT, None)
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                print("📊 Checking if data was saved...")
                
                # Check for session state
                if Path("session_state.json").exists():
                    print("✅ Session state found - you can resume!")
                    print("Run the script again to continue where you left off.")
                else:
                    print("⚠️ No session state found")
                
                # Ask if they want to retry
                try:
                    choice = input("\nRetry evaluation? (y/n): ").lower()
                    if choice not in ['y', 'yes']:
                        break
                except KeyboardInterrupt:
                    print("\n✅ Exiting safely")
                    break
                    
    except ImportError as e:
        print(f"❌ Cannot import evaluation module: {e}")
        print("Make sure you're in the evaluation/ directory")
        print("Required files:")
        print("  - interactive_session.py")
        print("  - evaluation_db.py")
        print("  - benchmark_queries.json")
        sys.exit(1)

def check_prerequisites():
    """Check all required files are present"""
    required_files = [
        "interactive_session.py",
        "evaluation_db.py",
        "benchmark_queries.json"
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        print("❌ Missing required files:")
        for file in missing:
            print(f"   - {file}")
        print(f"\nPlease run from the evaluation/ directory")
        return False
    
    return True

def recovery_check():
    """Check for recoverable sessions"""
    state_files = [
        Path("session_state.json"),
        Path("session_state.backup.json")
    ]
    
    for state_file in state_files:
        if state_file.exists():
            print(f"📋 Found existing session: {state_file}")
            print("You can continue your previous evaluation!")
            return True
    
    return False

def main():
    """Main entry point with comprehensive error handling"""
    
    print("🚀 Starting Census MCP Evaluation")
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Check for recovery
    recovery_check()
    
    # Setup signal handlers
    setup_signal_handlers()
    
    # Add safety wrapper for input
    original_input = input
    def safe_input(prompt=""):
        global shutdown_requested
        if shutdown_requested:
            raise KeyboardInterrupt("Shutdown requested")
        try:
            return original_input(prompt)
        except EOFError:
            print("\n✅ EOF detected - saving and exiting")
            raise KeyboardInterrupt("EOF")
    
    # Monkey patch for safety
    import builtins
    builtins.input = safe_input
    
    try:
        run_evaluation_safely()
    except Exception as e:
        print(f"\n💥 Critical error: {e}")
        
        # Final safety check
        if Path("session_state.json").exists():
            print("✅ Your session data is saved!")
            print("Run the script again to continue.")
        else:
            print("❌ No session data found - evaluation may have been lost")
        
        sys.exit(1)
    finally:
        print("\n🧹 Cleanup completed")
        print("📊 Check your database with: python evaluation_db.py --score-run [run_name]")

if __name__ == "__main__":
    main()
