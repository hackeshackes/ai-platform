import React, { useState } from 'react';
import { NodeCategory, NodeData } from './types';
import { Draggable } from './Draggable';

interface NodePanelProps {
  categories?: NodeCategory[];
  onNodeDragStart?: (nodeData: NodeData) => void;
}

// Default preset nodes
const defaultCategories: NodeCategory[] = [
  {
    id: 'agents',
    name: 'Agents',
    icon: '🤖',
    nodes: [
      {
        id: 'agent-chat',
        type: 'agent',
        label: 'Chat Agent',
        description: 'Conversational AI agent',
        icon: '💬',
        config: { model: 'gpt-4', temperature: 0.7 },
      },
      {
        id: 'agent-coding',
        type: 'agent',
        label: 'Coding Agent',
        description: 'Code generation and analysis',
        icon: '💻',
        config: { model: 'claude-code', temperature: 0.2 },
      },
      {
        id: 'agent-research',
        type: 'agent',
        label: 'Research Agent',
        description: 'Web search and data gathering',
        icon: '🔍',
        config: { searchDepth: 5 },
      },
    ],
  },
  {
    id: 'pipelines',
    name: 'Pipelines',
    icon: '🔄',
    nodes: [
      {
        id: 'pipeline-sequential',
        type: 'pipeline',
        label: 'Sequential Pipeline',
        description: 'Execute nodes in sequence',
        icon: '📊',
        config: { mode: 'sequential' },
      },
      {
        id: 'pipeline-parallel',
        type: 'pipeline',
        label: 'Parallel Pipeline',
        description: 'Execute nodes concurrently',
        icon: '⚡',
        config: { mode: 'parallel' },
      },
      {
        id: 'pipeline-conditional',
        type: 'pipeline',
        label: 'Conditional Pipeline',
        description: 'Branch based on conditions',
        icon: '🔀',
        config: { mode: 'conditional' },
      },
    ],
  },
  {
    id: 'custom',
    name: 'Custom',
    icon: '✨',
    nodes: [
      {
        id: 'custom-http',
        type: 'custom',
        label: 'HTTP Request',
        description: 'Make API calls',
        icon: '🌐',
        config: { method: 'GET', timeout: 30000 },
      },
      {
        id: 'custom-transform',
        type: 'custom',
        label: 'Data Transform',
        description: 'Transform and filter data',
        icon: '🔧',
        config: { function: 'identity' },
      },
      {
        id: 'custom-storage',
        type: 'custom',
        label: 'Storage',
        description: 'Save or load data',
        icon: '💾',
        config: { provider: 'local' },
      },
    ],
  },
];

export const NodePanel: React.FC<NodePanelProps> = ({
  categories = defaultCategories,
  onNodeDragStart,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(categories.map((c) => c.id))
  );

  const toggleCategory = (categoryId: string) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(categoryId)) {
        next.delete(categoryId);
      } else {
        next.add(categoryId);
      }
      return next;
    });
  };

  const handleDragStart = (nodeData: NodeData) => {
    onNodeDragStart?.(nodeData);
  };

  const filteredCategories = categories.map((category) => ({
    ...category,
    nodes: category.nodes.filter(
      (node) =>
        node.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
        node.description?.toLowerCase().includes(searchQuery.toLowerCase())
    ),
  })).filter((category) => category.nodes.length > 0);

  return (
    <div className="node-panel">
      <div className="node-panel-header">
        <h2>Components</h2>
        <input
          type="text"
          placeholder="Search nodes..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          style={{
            width: '100%',
            marginTop: '12px',
            padding: '8px 12px',
            border: '1px solid #e0e0e0',
            borderRadius: '6px',
            fontSize: '13px',
          }}
        />
      </div>

      <div style={{ flex: 1, overflow: 'auto' }}>
        {filteredCategories.map((category) => (
          <div key={category.id} className="node-category">
            <div
              className="node-category-header"
              onClick={() => toggleCategory(category.id)}
              style={{ cursor: 'pointer' }}
            >
              <span>{category.icon}</span>
              <span>{category.name}</span>
              <span style={{ marginLeft: 'auto', fontSize: '12px' }}>
                {expandedCategories.has(category.id) ? '▼' : '▶'}
              </span>
            </div>

            {expandedCategories.has(category.id) && (
              <div>
                {category.nodes.map((node) => (
                  <Draggable
                    key={node.id}
                    onDragStart={() => handleDragStart(node)}
                  >
                    <div
                      className="node-item"
                      data-drag-data={JSON.stringify(node)}
                      onDragStart={(e) => {
                        e.dataTransfer.setData(
                          'application/json',
                          JSON.stringify(node)
                        );
                        e.dataTransfer.effectAllowed = 'copy';
                      }}
                      draggable={!('ontouchstart' in window)}
                    >
                      <div className="node-item-icon">{node.icon || '📦'}</div>
                      <div className="node-item-info">
                        <div className="node-item-label">{node.label}</div>
                        {node.description && (
                          <div className="node-item-desc">{node.description}</div>
                        )}
                      </div>
                    </div>
                  </Draggable>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default NodePanel;
