/**
 * SmartDoc AI 全局布局
 *
 * 职责：
 *   - 顶部导航栏（Logo + 菜单切换）
 *   - 子路由渲染（Outlet）
 *   - 跨路由传递 URL 参数（导航切换时保留 ?conversation=xxx 参数）
 */
import React, { useCallback, useEffect, useRef } from 'react';
import { Outlet, useLocation, useNavigate } from 'umi';
import { Layout, Menu } from 'antd';
import type { MenuProps } from 'antd';
import { MessageOutlined, HistoryOutlined, FileTextOutlined } from '@ant-design/icons';
import { cancelActiveUpload } from '@/services/uploadSession';
import styles from './index.less';

const { Header, Content } = Layout;

type MenuItem = Required<MenuProps>['items'][number];

/** 顶部导航菜单项 */
const menuItems: MenuItem[] = [
  {
    key: '/chat',                        // 路由路径
    icon: <MessageOutlined />,
    label: '智能问答',
  },
  {
    key: '/history',
    icon: <HistoryOutlined />,
    label: '历史记录',
  },
];

const AppLayout: React.FC = () => {
  /** 当前路由信息（包含 pathname） */
  const location = useLocation();
  /** UmiJS 路由跳转方法 */
  const navigate = useNavigate();
  const prevPathRef = useRef(location.pathname);

  /** 离开 /chat 路由时中止上传（Layout 不卸载，比 Chat 内 listen 可靠） */
  useEffect(() => {
    const prev = prevPathRef.current;
    const curr = location.pathname;
    prevPathRef.current = curr;
    if (prev.startsWith('/chat') && !curr.startsWith('/chat')) {
      cancelActiveUpload('layout-route-leave');
    }
  }, [location.pathname]);

  /** 刷新 / 关闭标签页（仅 Layout 监听即可，不随 Chat 卸载失效） */
  useEffect(() => {
    const onPageUnload = () => cancelActiveUpload('page-unload');
    window.addEventListener('pagehide', onPageUnload);
    window.addEventListener('beforeunload', onPageUnload);
    return () => {
      window.removeEventListener('pagehide', onPageUnload);
      window.removeEventListener('beforeunload', onPageUnload);
    };
  }, []);

  /** 根据当前路径高亮对应菜单项 */
  const selectedKey: string = location.pathname.startsWith('/history')
    ? '/history'
    : '/chat';

  /** 点击 Logo 回到首页 */
  const handleLogoClick = useCallback((): void => {
    window.location.href = '/chat';
  }, [navigate]);

  /**
   * 导航菜单点击
   * ★ 核心：切换路由时保留当前 URL 的 ?conversation=xxx 参数
   *    这样从聊天页切到历史页再切回来，地址栏参数不丢，聊天页恢复正常
   */
  const handleMenuClick: MenuProps['onClick'] = useCallback(
    ({ key }: { key: string }): void => {
      if (key !== '/chat') {
        cancelActiveUpload('menu-nav');
      }
      const params = new URLSearchParams(window.location.search);
      const convId = params.get('conversation');
      navigate({ pathname: key, search: convId ? `?conversation=${convId}` : '' });
    },
    [navigate],
  );

  return (
    <Layout className={styles.root}>
      {/* 顶部导航栏 */}
      <Header className={styles.header}>
        <div className={styles.logo} onClick={handleLogoClick}>
          <FileTextOutlined className={styles.logoIcon} />
          <span className={styles.logoText}>SmartDoc AI</span>
        </div>
        <Menu
          theme="dark"
          mode="horizontal"
          selectedKeys={[selectedKey]}
          items={menuItems}
          onClick={handleMenuClick}
          className={styles.navMenu}
        />
      </Header>

      {/* 子页面渲染区 */}
      <Content className={styles.content}>
        <Outlet />
      </Content>
    </Layout>
  );
};

export default AppLayout;
