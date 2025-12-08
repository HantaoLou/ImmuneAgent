from langchain_mcp_adapters.client import (
    Connection,
    SSEConnection,
    StdioConnection,
    StreamableHttpConnection,
)
from pydantic import BaseModel


class ApplicationConfig(BaseModel):
    mcp_servers: dict[str, dict] = {
        "af3": {
            "transport": "sse",
            "url": "http://localhost:8084/sse",
            "timeout": 36000,
            "sse_read_timeout": 36000,
            "session_kwargs": {},
        },
        "imm": {
            "transport": "sse",
            "url": "http://117.148.176.36:8085/sse",
            "timeout": 120,
            "sse_read_timeout": 120,
            "session_kwargs": {},
        },
        "fdg": {
            "transport": "sse",
            "url": "http://117.148.176.36:8080/sse",
            "timeout": 120,
            "sse_read_timeout": 120,
            "session_kwargs": {},
        },
        "metabcr": {
            "transport": "sse",
            "url": "http://localhost:8082/sse",
            "timeout": 36000,
            "sse_read_timeout": 36000,
            "session_kwargs": {},
        },
        "airr": {
            "transport": "sse",
            "url": "http://localhost:8083/sse",
            "timeout": 36000,
            "sse_read_timeout": 36000,
            "session_kwargs": {},
        },
        "af3": {
            "transport": "sse",
            "url": "http://localhost:8084/sse",
            "timeout": 36000,
            "sse_read_timeout": 36000,
            "session_kwargs": {},
        },
        "imm": {
            "transport": "sse",
            "url": "http://localhost:8081/sse",
            "timeout": 36000,
            "sse_read_timeout": 36000,
            "session_kwargs": {},
        },
        "anarci": {
            "transport": "sse",
            "url": "http://localhost:8086/sse",
            "timeout": 36000,
            "sse_read_timeout": 36000,
            "session_kwargs": {},
        },
        "geo": {
            "transport": "sse",
            "url": "http://localhost:8087/sse",
            "timeout": 36000,
            "sse_read_timeout": 36000,
            "session_kwargs": {},
        },
        "lgblast": {
            "transport": "sse",
            "url": "http://localhost:8088/sse",
            "timeout": 36000,
            "sse_read_timeout": 36000,
            "session_kwargs": {},
        },
        "oas": {
            "transport": "sse",
            "url": "http://localhost:8089/sse",
            "timeout": 36000,
            "sse_read_timeout": 36000,
            "session_kwargs": {},
        },
        "bioinformatics": {
            "transport": "sse",
            "url": "http://localhost:8090/sse",
            "timeout": 36000,
            "sse_read_timeout": 36000,
            "session_kwargs": {},
        },
        "annotation": {
            "transport": "sse",
            "url": "http://localhost:8091/sse",
            "timeout": 36000,
            "sse_read_timeout": 36000,
            "session_kwargs": {},
        },
        "bcell": {
            "transport": "sse",
            "url": "http://localhost:8092/sse",
            "timeout": 36000,
            "sse_read_timeout": 36000,
            "session_kwargs": {},
        },
        "communication": {
            "transport": "sse",
            "url": "http://localhost:8093/sse",
            "timeout": 36000,
            "sse_read_timeout": 36000,
            "session_kwargs": {},
        },
        "multimodal": {
            "transport": "sse",
            "url": "http://localhost:8094/sse",
            "timeout": 36000,
            "sse_read_timeout": 36000,
            "session_kwargs": {},
        },
        "scrna": {
            "transport": "sse",
            "url": "http://localhost:8095/sse",
            "timeout": 36000,
            "sse_read_timeout": 36000,
            "session_kwargs": {},
        },
        "sabdab":{
            "transport": "sse",
            "url": "http://localhost:8096/sse",
            "timeout": 36000,
            "sse_read_timeout": 36000,
            "session_kwargs": {},
        },
        "scrna":{
            "transport": "sse",
            "url": "http://localhost:8092/sse",
            "timeout": 36000,
            "sse_read_timeout": 36000,
            "session_kwargs": {},
        },
        "ribonn":{
            "transport": "sse",
            "url": "http://localhost:8105/sse",
            "timeout": 36000,
            "sse_read_timeout": 36000,
            "session_kwargs": {},
        },
        "gemoRNA":{
            "transport": "sse",
            "url": "http://localhost:8106/sse",
            "timeout": 36000,
            "sse_read_timeout": 36000,
            "session_kwargs": {},
        },
        "CodonTransformer":{
            "transport": "sse",
            "url": "http://localhost:8107/sse",
            "timeout": 36000,
            "sse_read_timeout": 36000,
            "session_kwargs": {},
        },
        "rinalmo":{
            "transport": "sse",
            "url": "http://localhost:8108/sse",
            "timeout": 36000,
            "sse_read_timeout": 36000,
            "session_kwargs": {},
        },
        "flu": {
            "transport": "sse",
            "url": "http://117.148.176.36:8090/sse",
            "timeout": 36000,
            "sse_read_timeout": 36000,
            "session_kwargs": {},
        },
        "file_utils": {
            "transport": "sse",
            "url": "http://117.148.176.36:8091/sse",
            "timeout": 36000,
            "sse_read_timeout": 36000,
            "session_kwargs": {},
        },
        "integrateBcrData":{
            "transport": "sse",
            "url": "http://117.148.176.36:8092/sse",
            "timeout": 36000,
            "sse_read_timeout": 36000,
            "session_kwargs": {},
        },
    }

    @staticmethod
    def get_instance() -> "ApplicationConfig":
        return ApplicationConfig()

    def get_mcp_servers(self) -> dict[str, Connection]:
        res = dict()
        for name, c in self.mcp_servers.items():
            transport = c["transport"]
            if transport == "stdio":
                res[name] = StdioConnection(**c)
            elif transport == "sse":
                res[name] = SSEConnection(**c)
            elif transport == "streamable_http":
                res[name] = StreamableHttpConnection(**c)
            else:
                raise ValueError(f"Unknown transport type: {transport}")
        return res


if __name__ == "__main__":
    c = ApplicationConfig.get_instance()
    print(c.get_mcp_servers())
