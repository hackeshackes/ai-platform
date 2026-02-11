/**
 * V12 API 统一索引
 * 导出所有v12模块的API客户端
 */

// 气候模型 API
export * from './climate';

// 生物模拟 API
export * from './bio';

// 宇宙模拟 API
export * from './cosmos';

// 深空探测 API
export * from './deepspace';

// 量子模块 API
export * from './quantum';

// 元学习 API
export * from './meta';

// 涌现能力 API
export * from './emergence';

// 跨域推理 API
export * from './crossdomain';

// 持续学习 API
export * from './continual';

// AIOps API
export * from './aiops';

// 调度系统 API
export * from './scheduler';

// 自愈系统 API
export * from './selfhealing';

// 自动化运维 API
export * from './automation';

// 性能优化 API
export * from './performance';

/**
 * V12 API 使用说明
 * 
 * 导入示例:
 * import { climate, bio, cosmos } from '@/api/v12';
 * 
 * 或单独导入:
 * import { ClimateAPI } from '@/api/v12/climate';
 */

// API 基础配置
export const V12_API_BASE = '/api/v12';

// API 版本信息
export const V12_API_VERSION = '1.0.0';
