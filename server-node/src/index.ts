import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import multer from 'multer';
import uploadRoutes from './routes/upload';
import chatRoutes from './routes/chat';
import conversationRoutes from './routes/conversations';
import { startReconciliationScheduler } from './services/reconcile';
import { sendJsonError, toClientError } from './utils/clientError';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// 注册路由
app.use('/api/upload', uploadRoutes);
app.use('/api/chat', chatRoutes);
app.use('/api/conversations', conversationRoutes);

// 健康检查
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', service: 'node-gateway' });
});

/** 统一错误处理：Multer 等中间件错误 → 纯中文文案 */
app.use((err: unknown, _req: express.Request, res: express.Response, next: express.NextFunction) => {
  if (res.headersSent) {
    next(err);
    return;
  }
  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      sendJsonError(res, 400, '文件大小不能超过 20MB');
      return;
    }
    sendJsonError(res, 400, '文件上传失败');
    return;
  }
  const msg = err instanceof Error ? err.message : '';
  if (msg.includes('仅支持 PDF 和 TXT')) {
    sendJsonError(res, 400, msg);
    return;
  }
  console.error('未捕获错误:', err);
  sendJsonError(res, 500, toClientError(err));
});

app.listen(PORT, () => {
  console.log(`Node gateway running on http://localhost:${PORT}`);
  startReconciliationScheduler();
});