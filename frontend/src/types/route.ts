/**
 * 路由类型定义
 */

// 路由配置
export interface RouteConfig {
  path: string
  component: React.ComponentType<any>
  name: string
  icon?: string
  children?: RouteConfig[]
  permissions?: string[]
  hidden?: boolean
}

// 面包屑配置
export interface BreadcrumbConfig {
  title: string
  path?: string
}

// 路由守卫配置
export interface GuardConfig {
  requiresAuth: boolean
  requiredPermissions?: string[]
  redirectTo?: string
}

// 菜单项类型
export interface MenuItem {
  key: string
  label: string
  icon?: React.ReactNode
  children?: MenuItem[]
  type?: 'divider'
  hidden?: boolean
}

// v12 模块分组
export type V12ModuleGroup = 
  | 'democratization'    // AI民主化
  | 'hyperautomation'    // 超自动化
  | 'superintelligence'  // 超级智能
  | 'quantum'           // 量子AI
  | 'cosmos'            // 宇宙级AI
