from typing import Any
from mcp.server.fastmcp import FastMCP
import http.client
import json

mindv = FastMCP("mindverse")

url = "localhost:8002"
path = "/api/kernel2/chat"

messages =[]

@mindv.tool()
async def get_response(query:str) -> str | None | Any:
    """
    MCP server stores user preferences and memory. This method can be called when answering questions about user's own preferences or memory.

    Args:
        query (str): Query about user preferences or personal information

    Returns:
        str | None | Any: Response based on user's personal data
    """

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    messages.append({"role": "user", "content": query})

    data={
        "messages": messages,
        "stream": True
    }

    conn = http.client.HTTPConnection(url)

    # Send the POST request
    conn.request("POST", path, body=json.dumps(data), headers=headers)

    # Get the response
    response = conn.getresponse()
    full_content=""

    for line in response:
        if line:
            decoded_line = line.decode('utf-8').strip()
            if decoded_line == 'data: [DONE]':
                break
            if decoded_line.startswith('data: '):
                try:
                    json_str = decoded_line[6:]
                    chunk = json.loads(json_str)
                    content = chunk['choices'][0]['delta'].get('content', '')
                    if content:
                        full_content+=content
                except json.JSONDecodeError:
                    pass

    conn.close()
    if full_content:
        messages.append({"role": "system", "content": full_content})
        return full_content
    else:
        return None


if __name__ == "__main__":
    # Initialize and run the server
    mindv.run(transport='stdio')
