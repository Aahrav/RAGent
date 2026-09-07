import httpx
import time
import asyncio
import json

BASE_URL = "http://localhost:8000"
TEST_QUERY = "Who is the ceo of meta?"
QUERY_2 = "Ceo of meta is?"

def print_header(title: str):
    print(f"\n{'='*50}\n{title}\n{'='*50}")

async def test_streaming_and_caching():
    print_header("1. Testing Streaming Response (First Query - Cache Miss)")
    
    start_time = time.time()
    # No timeout, as the local server can take a long time under load
    timeout = httpx.Timeout(None)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # We use httpx to consume the SSE stream
        async with client.stream("POST", f"{BASE_URL}/chat/stream", json={"query": TEST_QUERY}) as response:
            if response.status_code != 200:
                print(f"Error: {response.status_code} - {await response.aread()}")
                return
                
            print("Assistant: ", end="", flush=True)
            
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                    
                data_str = line[6:] # Strip "data: "
                data = json.loads(data_str)
                
                if "chunk" in data:
                    print(data["chunk"], end="", flush=True)
                elif "metadata" in data:
                    print(f"\n\n[Metadata Received] Latency: {data['metadata']['latency_ms']}ms, "
                          f"Confidence: {data['metadata']['confidence']}, "
                          f"Citations: {len(data['metadata']['citations'])}")
                          
    duration_1 = time.time() - start_time
    print(f"\nFirst query took {duration_1:.2f} seconds.")
    
    
    print_header("2. Testing Semantic Caching (Same Query - Cache Hit)")
    print("Sending the exact same query. Watch how fast it returns...")
    
    start_time = time.time()
    # No timeout
    timeout = httpx.Timeout(None)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", f"{BASE_URL}/chat/stream", json={"query": QUERY_2}) as response:
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                return
                
            print("Assistant: ", end="", flush=True)
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if "chunk" in data:
                        print(data["chunk"], end="", flush=True)
                    elif "metadata" in data:
                        tools = data['metadata'].get('tools_used', [])
                        print(f"\n\n[Metadata Received] Cache Hit! Tools used: {tools}, "
                              f"Latency recorded: {data['metadata']['latency_ms']}ms")
                              
    duration_2 = time.time() - start_time
    print(f"\nSecond query took {duration_2:.4f} seconds.")
    print(f"Speedup: {duration_1 / duration_2:.1f}x faster!")

async def make_request(client, i):
    response = await client.post(f"{BASE_URL}/chat", json={"query": f"Query {i}"})
    return response.status_code

async def test_rate_limiting():
    print_header("3. Testing Rate Limiting (Spamming API)")
    print("The default rate limit is 60 requests per minute. We will send 10 fast requests.")
    
    # No timeout, to allow the server to process the heavy CPU load
    timeout = httpx.Timeout(None)
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [make_request(client, i) for i in range(10)]

        
        print("Sending 65 concurrent requests (this may take a moment as the first 60 get queued)...")
        statuses = await asyncio.gather(*tasks, return_exceptions=True)
        
        successes = 0
        rate_limits = 0
        others = 0
        
        for status in statuses:
            if isinstance(status, Exception):
                others += 1
            elif status == 200:
                successes += 1
            elif status == 429:
                rate_limits += 1
            else:
                others += 1
        
        print(f"Results:")
        print(f"  200 OK (Allowed): {successes}")
        print(f"  429 Too Many Requests (Blocked): {rate_limits}")
        print(f"  Other errors: {others}")
        
        if rate_limits > 0:
            print("\n✅ Rate limiter is working perfectly! It blocked the excess requests.")
        else:
            print("\n❌ Rate limiter didn't trigger. Are rate limits enabled in config?")

async def main():
    try:
        await test_streaming_and_caching()
        await test_rate_limiting()
    except httpx.ConnectError:
        print("❌ Could not connect to the API. Is the FastAPI server running?")
        print("Run `uv run uvicorn src.main:app` first.")

if __name__ == "__main__":
    asyncio.run(main())
