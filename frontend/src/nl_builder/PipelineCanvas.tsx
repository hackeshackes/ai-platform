import React, { useRef, useState, useCallback, useEffect, useMemo } from 'react';
import { PipelineNode, Connection, Viewport, NodePosition } from './types';
import { Draggable } from './Draggable';

interface PipelineCanvasProps {
  nodes: PipelineNode[];
  connections: Connection[];
  viewport: Viewport;
  selectedNodeId?: string;
  onNodesChange: (nodes: PipelineNode[]) => void;
  onConnectionsChange: (connections: Connection[]) => void;
  onViewportChange: (viewport: Viewport) => void;
  onNodeSelect: (nodeId?: string) => void;
  onNodeDrop: (nodeData: unknown, position: NodePosition) => void;
  onCanvasClick: () => void;
}

export const PipelineCanvas: React.FC<PipelineCanvasProps> = ({
  nodes,
  connections,
  viewport,
  selectedNodeId,
  onNodesChange,
  onConnectionsChange,
  onViewportChange,
  onNodeSelect,
  onNodeDrop,
  onCanvasClick,
}) => {
  const canvasRef = useRef<HTMLDivElement>(null);
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState<NodePosition>({ x: 0, y: 0 });
  const [connectingNode, setConnectingNode] = useState<{
    nodeId: string;
    sourceHandle?: string;
    startPos: NodePosition;
  } | null>(null);
  const [connectionPreview, setConnectionPreview] = useState<{
    startPos: NodePosition;
    endPos: NodePosition;
  } | null>(null);

  // Convert screen coordinates to canvas coordinates
  const screenToCanvas = useCallback(
    (screenX: number, screenY: number): NodePosition => {
      return {
        x: (screenX - viewport.x) / viewport.zoom,
        y: (screenY - viewport.y) / viewport.zoom,
      };
    },
    [viewport.x, viewport.y, viewport.zoom]
  );

  // Convert canvas coordinates to screen coordinates
  const canvasToScreen = useCallback(
    (canvasX: number, canvasY: number): NodePosition => {
      return {
        x: canvasX * viewport.zoom + viewport.x,
        y: canvasY * viewport.zoom + viewport.y,
      };
    },
    [viewport.x, viewport.y, viewport.zoom]
  );

  // Handle mouse wheel for zoom
  const handleWheel = useCallback(
    (event: React.WheelEvent) => {
      if (event.ctrlKey || event.metaKey) {
        event.preventDefault();
        const delta = event.deltaY > 0 ? 0.9 : 1.1;
        const newZoom = Math.min(Math.max(viewport.zoom * delta, 0.1), 5);
        
        // Zoom towards mouse position
        const rect = canvasRef.current?.getBoundingClientRect();
        if (rect) {
          const mouseX = event.clientX - rect.left;
          const mouseY = event.clientY - rect.top;
          
          const newX = mouseX - (mouseX - viewport.x) * (newZoom / viewport.zoom);
          const newY = mouseY - (mouseY - viewport.y) * (newZoom / viewport.zoom);
          
          onViewportChange({ x: newX, y: newY, zoom: newZoom });
        }
      } else {
        // Pan with wheel
        onViewportChange({
          ...viewport,
          x: viewport.x - event.deltaX,
          y: viewport.y - event.deltaY,
        });
      }
    },
    [viewport, onViewportChange]
  );

  // Handle pan start (middle mouse or space+drag)
  const handlePanStart = useCallback(
    (event: React.MouseEvent) => {
      if (event.button === 1 || event.button === 0) {
        setIsPanning(true);
        setPanStart({ x: event.clientX, y: event.clientY });
      }
    },
    []
  );

  // Handle pan move
  useEffect(() => {
    if (!isPanning) return;

    const handleMouseMove = (event: MouseEvent) => {
      const deltaX = event.clientX - panStart.x;
      const deltaY = event.clientY - panStart.y;
      
      onViewportChange({
        ...viewport,
        x: viewport.x + deltaX,
        y: viewport.y + deltaY,
      });
      
      setPanStart({ x: event.clientX, y: event.clientY });
    };

    const handleMouseUp = () => {
      setIsPanning(false);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isPanning, panStart, viewport, onViewportChange]);

  // Handle connection creation
  const handleConnectionStart = useCallback(
    (event: React.MouseEvent, nodeId: string, handleId: string) => {
      event.stopPropagation();
      
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;

      const startPos = screenToCanvas(event.clientX - rect.left, event.clientY - rect.top);
      
      setConnectingNode({ nodeId, sourceHandle: handleId, startPos });
    },
    [screenToCanvas]
  );

  // Handle connection preview
  useEffect(() => {
    if (!connectingNode) {
      setConnectionPreview(null);
      return;
    }

    const handleMouseMove = (event: MouseEvent) => {
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;

      const endPos = screenToCanvas(event.clientX - rect.left, event.clientY - rect.top);
      setConnectionPreview({
        startPos: connectingNode.startPos,
        endPos,
      });
    };

    document.addEventListener('mousemove', handleMouseMove);
    return () => document.removeEventListener('mousemove', handleMouseMove);
  }, [connectingNode, screenToCanvas]);

  // Handle connection end
  const handleConnectionEnd = useCallback(
    (event: React.MouseEvent) => {
      if (!connectingNode) return;

      // Check if dropped on a target handle
      const targetElement = document.elementFromPoint(
        event.clientX,
        event.clientY
      );
      
      const targetHandle = targetElement?.closest('.handle.target');
      
      if (targetHandle) {
        const targetNodeId = targetHandle.dataset.nodeId;
        const targetHandleId = targetHandle.dataset.handleId;
        
        if (targetNodeId && targetNodeId !== connectingNode.nodeId) {
          const newConnection: Connection = {
            id: `conn-${Date.now()}`,
            sourceId: connectingNode.nodeId,
            targetId: targetNodeId,
            sourceHandle: connectingNode.sourceHandle,
            targetHandle: targetHandleId,
          };
          onConnectionsChange([...connections, newConnection]);
        }
      }

      setConnectingNode(null);
      setConnectionPreview(null);
    },
    [connectingNode, connections, onConnectionsChange]
  );

  // Handle node drop from panel
  const handleNodeDrop = useCallback(
    (item: unknown, position: NodePosition) => {
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;

      const canvasPosition = screenToCanvas(
        position.x - rect.left,
        position.y - rect.top
      );
      
      onNodeDrop(item, canvasPosition);
    },
    [onNodeDrop, screenToCanvas]
  );

  // Render connection path
  const renderConnectionPath = useMemo(
    () => (conn: Connection) => {
      const sourceNode = nodes.find((n) => n.id === conn.sourceId);
      const targetNode = nodes.find((n) => n.id === conn.targetId);
      
      if (!sourceNode || !targetNode) return null;

      const startPos = canvasToScreen(
        sourceNode.position.x + (sourceNode.size?.width || 180),
        sourceNode.position.y + (sourceNode.size?.height || 80) / 2
      );
      const endPos = canvasToScreen(
        targetNode.position.x,
        targetNode.position.y + (targetNode.size?.height || 80) / 2
      );

      const controlOffset = Math.abs(startPos.x - endPos.x) * 0.5;
      
      return (
        <path
          key={conn.id}
          d={`M ${startPos.x} ${startPos.y} 
              C ${startPos.x + controlOffset} ${startPos.y},
                ${endPos.x - controlOffset} ${endPos.y},
                ${endPos.x} ${endPos.y}`}
          className={selectedNodeId === conn.id || selectedNodeId === conn.sourceId || selectedNodeId === conn.targetId ? 'selected' : ''}
        />
      );
    },
    [nodes, canvasToScreen, selectedNodeId]
  );

  // Render connection preview
  const renderConnectionPreview = () => {
    if (!connectionPreview) return null;

    const startScreen = canvasToScreen(connectionPreview.startPos.x, connectionPreview.startPos.y);
    const endScreen = canvasToScreen(connectionPreview.endPos.x, connectionPreview.endPos.y);
    
    const controlOffset = Math.abs(startScreen.x - endScreen.x) * 0.5;

    return (
      <path
        d={`M ${startScreen.x} ${startScreen.y} 
            C ${startScreen.x + controlOffset} ${startScreen.y},
              ${endScreen.x - controlOffset} ${endScreen.y},
              ${endScreen.x} ${endScreen.y}`}
        className="connection-preview"
      />
    );
  };

  return (
    <div
      ref={canvasRef}
      className="pipeline-canvas"
      onWheel={handleWheel}
      onMouseDown={handlePanStart}
      onMouseUp={handleConnectionEnd}
      onContextMenu={(e) => e.preventDefault()}
      style={{
        transform: `translate(${viewport.x}px, ${viewport.y}px) scale(${viewport.zoom})`,
        transformOrigin: '0 0',
      }}
    >
      {/* Drop zone for new nodes */}
      <Draggable onDrop={handleNodeDrop}>
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none',
          }}
        />
      </Draggable>

      {/* SVG Layer for connections */}
      <svg
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none',
          overflow: 'visible',
        }}
      >
        {connections.map(renderConnectionPath)}
        {renderConnectionPreview()}
      </svg>

      {/* Nodes */}
      {nodes.map((node) => (
        <div
          key={node.id}
          className={`pipeline-node ${selectedNodeId === node.id ? 'selected' : ''}`}
          style={{
            left: node.position.x,
            top: node.position.y,
            width: node.size.width,
            height: node.size.height,
          }}
          onClick={(e) => {
            e.stopPropagation();
            onNodeSelect(node.id);
          }}
        >
          <div className="pipeline-node-header">
            <div className="pipeline-node-icon">{node.data.icon || '📦'}</div>
            <div className="pipeline-node-title">{node.data.label}</div>
          </div>
          <div className="pipeline-node-body">
            <div className="pipeline-node-description">
              {node.data.description || 'No description'}
            </div>
          </div>

          {/* Connection handles */}
          <div
            className="handle source"
            data-node-id={node.id}
            data-handle-id="output"
            onMouseDown={(e) => handleConnectionStart(e, node.id, 'output')}
          />
          <div
            className="handle target"
            data-node-id={node.id}
            data-handle-id="input"
          />
        </div>
      ))}

      {/* Empty state */}
      {nodes.length === 0 && (
        <div className="empty-state">
          <div className="empty-state-icon">🎯</div>
          <div className="empty-state-title">Start Building</div>
          <div className="empty-state-desc">
            Drag nodes from the left panel to create your pipeline
          </div>
        </div>
      )}
    </div>
  );
};

export default PipelineCanvas;
