// V9 页面路由配置
import { AdaptiveLearning } from './pages/v9/AdaptiveLearning';
import { FederatedLearning } from './pages/v9/FederatedLearning';
import { DecisionEngine } from './pages/v9/DecisionEngine';

// V9 Routes
const v9Routes = [
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
  }
];

export default v9Routes;
