#!/usr/bin/env python3
"""
KV Events Listener
Simple ZMQ subscriber to listen for KV cache events from vllm.
"""

import argparse
import zmq
import msgspec
import msgspec.msgpack
from datetime import datetime


def import_kv_events():
    """Dynamically import KV events from vllm"""
    try:
        from vllm.distributed.kv_events import KVEventBatch
        return KVEventBatch
    except ImportError:
        print("Warning: Could not import KVEventBatch from vllm")
        print("Will try to decode as generic message")
        return None


def main():
    KVEventBatch = import_kv_events()
    
    parser = argparse.ArgumentParser(description="KV Events Listener")
    parser.add_argument("--endpoint", type=str, default="tcp://localhost:5555", 
                        help="ZMQ endpoint to connect to (default: tcp://localhost:5555)")
    parser.add_argument("--topic", type=str, default="kv-events", 
                        help="ZMQ topic to subscribe to (default: kv_events)")
    args = parser.parse_args()
    
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    
    # Configuration
    endpoint = args.endpoint
    topic = args.topic
    topic_bytes = topic.encode("utf-8")
    
    print(f"Connecting to {endpoint}...")
    socket.connect(endpoint)
    
    print(f"Subscribing to topic: {topic}")
    socket.setsockopt(zmq.SUBSCRIBE, topic_bytes)
    
    print("=" * 60)
    print("Listening for KV events... (Press Ctrl+C to stop)")
    print("=" * 60)
    
    # Set up decoder
    decoder = None
    if KVEventBatch:
        decoder = msgspec.msgpack.Decoder(type=KVEventBatch)
    
    print(f"✓ Endpoint: {endpoint}")
    print(f"✓ Topic: {topic}")
    print(f"✓ Socket type: SUB")
    print(f"✓ Decoder: {'Available' if decoder else 'Not available'}")
    print("=" * 60)
    print("Waiting for events...")

    event_count = 0
    
    try:
        while True:
            # Use poller with timeout to show that we're alive
            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            
            # Poll with timeout every 2 seconds to show liveness
            events = dict(poller.poll(2000))
            
            if not events:
                # No events received, just show a heartbeat
                continue
            
            # Receive multipart message
            print("\n" + "=" * 60)
            print("📨 Received message!")
            print("=" * 60)
            
            message_parts = socket.recv_multipart()
            print(f"Number of parts: {len(message_parts)}")
            for i, part in enumerate(message_parts):
                print(f"Part {i} ({len(part)} bytes): {part!r}")
            
            if len(message_parts) != 3:
                print(f"\n⚠️  Warning: Expected 3 parts, got {len(message_parts)}")
                continue
            
            topic_recv, seq_bytes, payload = message_parts
            
            assert topic_recv == topic_bytes, f"Unexpected topic: {topic_recv!r}"
            
            # Parse sequence number
            seq = int.from_bytes(seq_bytes, "big")
            print(f"Sequence: {seq}")
            
            event_count += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            
            print(f"\n[{event_count}] Received event at: {timestamp}")
            
            # Try to decode the payload
            if decoder:
                try:
                    event_batch = decoder.decode(payload)
                    print(f"\n✅ Decoded successfully!")
                    print(f"  Event type: {type(event_batch).__name__}")
                    print(f"  Timestamp: {event_batch.ts}")
                    print(f"  DP rank: {event_batch.data_parallel_rank}")
                    print(f"  Number of events: {len(event_batch.events)}")
                    
                    for i, event in enumerate(event_batch.events):
                        print(f"  Event {i+1}: {type(event).__name__}")
                        print(f"    {event}")
                except Exception as e:
                    print(f"\n❌ Failed to decode as KVEventBatch: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"\n  Raw payload: {payload!r}")
            else:
                print(f"\n  Raw payload: {payload!r}")
                    
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print(f"Stopped. Received {event_count} events total.")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()