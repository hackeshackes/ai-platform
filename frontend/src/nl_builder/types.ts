// Pipeline Builder Types

export interface NodePosition {
  x: number;
  y: number;
}

export interface NodeSize {
  width: number;
  height: number;
}

export interface NodeData {
  id: string;
  type: 'agent' | 'pipeline' | 'custom';
  label: string;
  description?: string;
  icon?: string;
  config?: Record<string, unknown>;
}

export interface PipelineNode {
  id: string;
  data: NodeData;
  position: NodePosition;
  size: NodeSize;
  selected?: boolean;
}

export interface Connection {
  id: string;
  sourceId: string;
  targetId: string;
  sourceHandle?: string;
  targetHandle?: string;
}

export interface Viewport {
  x: number;
  y: number;
  zoom: number;
}

export interface PipelineState {
  nodes: PipelineNode[];
  connections: Connection[];
  viewport: Viewport;
  selectedNodeId?: string;
  selectedConnectionId?: string;
}

export interface NodeCategory {
  id: string;
  name: string;
  icon: string;
  nodes: NodeData[];
}

export interface DragItem {
  type: 'node' | 'connection';
  data: unknown;
  offset?: NodePosition;
}

export interface CanvasEvent {
  type: 'node_drag' | 'node_drag_start' | 'node_drag_end' | 
        'connection_create' | 'connection_complete' | 'connection_cancel' |
        'viewport_change' | 'node_select' | 'canvas_click' | 'node_drop';
  payload: unknown;
  timestamp: number;
}
