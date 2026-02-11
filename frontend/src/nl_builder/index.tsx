import React, { useState, useCallback, useEffect } from 'react';
import { PipelineState, PipelineNode, Connection, Viewport, NodeData, NodePosition } from './types';
import { NodePanel } from './NodePanel';
import { PipelineCanvas } from './PipelineCanvas';
import { Preview } from './Preview';
import { Toolbar } from './Toolbar';
import './styles.css';

// Initial state
const initialState: PipelineState = {
  nodes: [],
  connections: [],
  viewport: { x: 0, y: 0, zoom: 1 },
  selectedNodeId: undefined,
  selectedConnectionId: undefined,
};

// History for undo/redo
interface HistoryEntry {
  state: PipelineState;
}

export const NLBuilder: React.FC = () => {
  const [pipelineState, setPipelineState] = useState<PipelineState>(initialState);
  const [history, setHistory] = useState<HistoryEntry[]>([initialState]);
  const [historyIndex, setHistoryIndex] = useState(0);

  // Push state to history
  const pushHistory = useCallback((newState: PipelineState) => {
    setHistory((prev) => {
      const trimmed = prev.slice(0, historyIndex + 1);
      return [...trimmed, newState];
    });
    setHistoryIndex((prev) => prev + 1);
  }, [historyIndex]);

  // Undo
  const handleUndo = useCallback(() => {
    if (historyIndex > 0) {
      setHistoryIndex((prev) => prev - 1);
      setPipelineState(history[historyIndex - 1].state);
    }
  }, [history, historyIndex]);

  // Redo
  const handleRedo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      setHistoryIndex((prev) => prev + 1);
      setPipelineState(history[historyIndex + 1].state);
    }
  }, [history, historyIndex]);

  // Save
  const handleSave = useCallback((state: PipelineState) => {
    console.log('Saving pipeline state:', state);
    // In production, save to server/database
  }, []);

  // Load
  const handleLoad = useCallback((state: PipelineState) => {
    setPipelineState(state);
    pushHistory(state);
  }, [pushHistory]);

  // Export
  const handleExport = useCallback((state: PipelineState) => {
    console.log('Exporting pipeline:', state);
    // Generate and download code
  }, []);

  // Import
  const handleImport = useCallback((file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const state = JSON.parse(e.target?.result as string) as PipelineState;
        handleLoad(state);
      } catch (error) {
        console.error('Failed to import file:', error);
        alert('Failed to import file. Please check the format.');
      }
    };
    reader.readAsText(file);
  }, [handleLoad]);

  // Handle nodes change
  const handleNodesChange = useCallback((nodes: PipelineNode[]) => {
    const newState = { ...pipelineState, nodes };
    setPipelineState(newState);
    pushHistory(newState);
  }, [pipelineState, pushHistory]);

  // Handle connections change
  const handleConnectionsChange = useCallback((connections: Connection[]) => {
    const newState = { ...pipelineState, connections };
    setPipelineState(newState);
    pushHistory(newState);
  }, [pipelineState, pushHistory]);

  // Handle viewport change
  const handleViewportChange = useCallback((viewport: Viewport) => {
    setPipelineState((prev) => ({ ...prev, viewport }));
  }, []);

  // Handle node select
  const handleNodeSelect = useCallback((nodeId?: string) => {
    setPipelineState((prev) => ({
      ...prev,
      selectedNodeId: nodeId,
      selectedConnectionId: undefined,
    }));
  }, []);

  // Handle canvas click
  const handleCanvasClick = useCallback(() => {
    setPipelineState((prev) => ({
      ...prev,
      selectedNodeId: undefined,
      selectedConnectionId: undefined,
    }));
  }, []);

  // Handle node drop
  const handleNodeDrop = useCallback((nodeData: unknown, position: NodePosition) => {
    const data = nodeData as NodeData;
    
    const newNode: PipelineNode = {
      id: `${data.type}-${Date.now()}`,
      data: {
        ...data,
        id: `${data.type}-${Date.now()}`,
      },
      position: {
        x: position.x - 90, // Center the node
        y: position.y - 40,
      },
      size: {
        width: 180,
        height: 80,
      },
    };

    handleNodesChange([...pipelineState.nodes, newNode]);
  }, [pipelineState.nodes, handleNodesChange]);

  // Handle node position change
  const handleNodePositionChange = useCallback((nodeId: string, position: NodePosition) => {
    const newNodes = pipelineState.nodes.map((node) =>
      node.id === nodeId ? { ...node, position } : node
    );
    handleNodesChange(newNodes);
  }, [pipelineState.nodes, handleNodesChange]);

  // Delete selected
  const handleDeleteSelected = useCallback(() => {
    let newNodes = pipelineState.nodes;
    let newConnections = pipelineState.connections;

    if (pipelineState.selectedNodeId) {
      newNodes = newNodes.filter((n) => n.id !== pipelineState.selectedNodeId);
      newConnections = newConnections.filter(
        (c) => c.sourceId !== pipelineState.selectedNodeId && c.targetId !== pipelineState.selectedNodeId
      );
    }

    if (pipelineState.selectedConnectionId) {
      newConnections = newConnections.filter(
        (c) => c.id !== pipelineState.selectedConnectionId
      );
    }

    setPipelineState((prev) => ({
      ...prev,
      selectedNodeId: undefined,
      selectedConnectionId: undefined,
    }));

    handleNodesChange(newNodes);
    handleConnectionsChange(newConnections);
  }, [
    pipelineState.selectedNodeId,
    pipelineState.selectedConnectionId,
    pipelineState.nodes,
    pipelineState.connections,
    handleNodesChange,
    handleConnectionsChange,
  ]);

  // Zoom controls
  const handleZoomIn = useCallback(() => {
    const newZoom = Math.min(pipelineState.viewport.zoom * 1.2, 5);
    handleViewportChange({ ...pipelineState.viewport, zoom: newZoom });
  }, [pipelineState.viewport, handleViewportChange]);

  const handleZoomOut = useCallback(() => {
    const newZoom = Math.max(pipelineState.viewport.zoom / 1.2, 0.1);
    handleViewportChange({ ...pipelineState.viewport, zoom: newZoom });
  }, [pipelineState.viewport, handleViewportChange]);

  const handleZoomReset = useCallback(() => {
    handleViewportChange({ x: 0, y: 0, zoom: 1 });
  }, [handleViewportChange]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Delete' || event.key === 'Backspace') {
        if (
          document.activeElement?.tagName !== 'INPUT' &&
          document.activeElement?.tagName !== 'TEXTAREA'
        ) {
          handleDeleteSelected();
        }
      }

      if ((event.ctrlKey || event.metaKey) && event.key === 'z') {
        event.preventDefault();
        if (event.shiftKey) {
          handleRedo();
        } else {
          handleUndo();
        }
      }

      if ((event.ctrlKey || event.metaKey) && event.key === 'y') {
        event.preventDefault();
        handleRedo();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleUndo, handleRedo, handleDeleteSelected]);

  return (
    <div className="nl-builder">
      <NodePanel
        onNodeDragStart={(nodeData) => {
          console.log('Dragging node:', nodeData);
        }}
      />

      <div className="canvas-container">
        <Toolbar
          pipelineState={pipelineState}
          onSave={handleSave}
          onLoad={handleLoad}
          onExport={handleExport}
          onImport={handleImport}
          onUndo={handleUndo}
          onRedo={handleRedo}
          onDeleteSelected={handleDeleteSelected}
          onZoomIn={handleZoomIn}
          onZoomOut={handleZoomOut}
          onZoomReset={handleZoomReset}
          canUndo={historyIndex > 0}
          canRedo={historyIndex < history.length - 1}
        />

        <PipelineCanvas
          nodes={pipelineState.nodes}
          connections={pipelineState.connections}
          viewport={pipelineState.viewport}
          selectedNodeId={pipelineState.selectedNodeId}
          onNodesChange={handleNodesChange}
          onConnectionsChange={handleConnectionsChange}
          onViewportChange={handleViewportChange}
          onNodeSelect={handleNodeSelect}
          onNodeDrop={handleNodeDrop}
          onCanvasClick={handleCanvasClick}
        />

        {/* Zoom controls */}
        <div className="zoom-controls">
          <button className="zoom-btn" onClick={handleZoomIn}>
            +
          </button>
          <div className="zoom-level">{Math.round(pipelineState.viewport.zoom * 100)}%</div>
          <button className="zoom-btn" onClick={handleZoomOut}>
            -
          </button>
        </div>
      </div>

      <Preview
        nodes={pipelineState.nodes}
        connections={pipelineState.connections}
      />
    </div>
  );
};

export default NLBuilder;
