import { defineConfig } from "umi";

export default defineConfig({
  /**
   * 前端路由表
   * / → 重定向到聊天页
   * /chat → 智能问答主页
   * /history → 历史会话列表
   */
  routes: [
    { path: "/", redirect: "/chat" },
    { path: "/chat", component: "chat/index" },
    { path: "/history", component: "history/index" },
  ],

  npmClient: 'yarn',

  /**
   * 开发代理：所有 /api 请求转发到 Node.js 网关 (Express, 端口 3000)
   * SSE 路由单独配置，避免 dev proxy 缓冲导致进度不实时
   */
  proxy: {
    '/api/upload/process': {
      target: 'http://localhost:3000',
      changeOrigin: true,
      onProxyRes: (proxyRes: { headers: Record<string, string> }) => {
        proxyRes.headers['cache-control'] = 'no-cache';
        proxyRes.headers['x-accel-buffering'] = 'no';
      },
    },
    '/api/chat/ask': {
      target: 'http://localhost:3000',
      changeOrigin: true,
      onProxyRes: (proxyRes: { headers: Record<string, string> }) => {
        proxyRes.headers['cache-control'] = 'no-cache';
        proxyRes.headers['x-accel-buffering'] = 'no';
      },
    },
    '/api': {
      target: 'http://localhost:3000',
      changeOrigin: true,
    },
  },

  utoopack: {},
});
