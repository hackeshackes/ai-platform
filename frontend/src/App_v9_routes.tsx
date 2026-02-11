// V9 Routes
import { AdaptiveLearning, FederatedLearning, DecisionEngine } from './pages/v9';

// 添加到routes数组中:
{
  path: '/v9/adaptive',
  element: <AdaptiveLearning />,
  icon: 'brain',
  label: '自适应学习'
},
{
  path: '/v9/federated',
  element: <FederatedLearning />,
  icon: 'share',
  label: '联邦学习'
},
{
  path: '/v9/decision',
  element: <DecisionEngine />,
  icon: 'target',
  label: '决策引擎'
},
