## Run frontend

Install latest [nodejs](https://nodejs.org/en/download) LTS version.

Change backend address in `vite.config.ts` and run.

```sh
# install dependencies
npm install
```

```sh
# to start debug server locally
npm run dev
```

## 框架
- [react](https://react.dev/) web 框架
- [ai-sdk](https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol) 前后端消息 stream 协议
- [llamaindex ui](https://ts.llamaindex.ai/docs/chat-ui) 聊天界面
- [ant design](https://ant.design/) 组件库
- [vite](https://vite.dev/) 前端开发工具


## 构建前端
```
mkdir /opt/antibody_gen/
npm run build
cp -r dist /opt/antibody_gen/
```
