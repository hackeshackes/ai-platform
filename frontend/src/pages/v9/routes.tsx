// V9 页面路由配置

export const v9Routes = [
  {
    path: '/v9/adaptive',
    name: 'adaptive-learning',
    label: '自适应学习',
    icon: 'brain',
    component: () => import('./AdaptiveLearning')
  },
  {
    path: '/v9/federated',
    name: 'federated-learning',
    label: '联邦学习',
    icon: 'share',
    component: () => import('./FederatedLearning')
  },
  {
    path: '/v9/decision',
    name: 'decision-engine',
    label: '决策引擎',
    icon: 'target',
    component: () => import('./DecisionEngine')
  }
];

// 添加到主路由配置中使用
export default v9Routes;
