import React, { useCallback, useRef, useState, useEffect } from 'react';
import { PipelineNode, NodePosition, DragItem } from './types';

interface DraggableProps {
  children: React.ReactNode;
  onDragStart?: (item: DragItem) => void;
  onDrag?: (position: NodePosition) => void;
  onDragEnd?: () => void;
  onDrop?: (item: DragItem, position: NodePosition) => void;
  disabled?: boolean;
}

export const Draggable: React.FC<DraggableProps> = ({
  children,
  onDragStart,
  onDrag,
  onDragEnd,
  onDrop,
  disabled = false,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [dragPosition, setDragPosition] = useState<NodePosition>({ x: 0, y: 0 });
  const itemRef = useRef<DragItem | null>(null);
  const startPosRef = useRef<NodePosition | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  const handleMouseDown = useCallback(
    (event: React.MouseEvent | React.TouchEvent) => {
      if (disabled) return;

      const clientX = 'touches' in event ? event.touches[0].clientX : event.clientX;
      const clientY = 'touches' in event ? event.touches[0].clientY : event.clientY;

      setIsDragging(true);
      startPosRef.current = { x: clientX, y: clientY };

      // Find the drag data from the element
      const target = event.currentTarget as HTMLElement;
      const dragData = target.dataset.dragData;
      
      if (dragData) {
        try {
          itemRef.current = JSON.parse(dragData);
          onDragStart?.(itemRef.current);
        } catch {
          // Invalid drag data
        }
      }

      // Prevent default to avoid text selection
      if ('preventDefault' in event) {
        event.preventDefault();
      }
    },
    [disabled, onDragStart]
  );

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (event: MouseEvent | TouchEvent) => {
      const clientX = 'touches' in event ? event.touches[0].clientX : event.clientX;
      const clientY = 'touches' in event ? event.touches[0].clientY : event.clientY;

      if (startPosRef.current) {
        const deltaX = clientX - startPosRef.current.x;
        const deltaY = clientY - startPosRef.current.y;

        setDragPosition({ x: deltaX, y: deltaY });

        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
        }

        animationFrameRef.current = requestAnimationFrame(() => {
          onDrag?.({ x: deltaX, y: deltaY });
        });
      }
    };

    const handleMouseUp = (event: MouseEvent | TouchEvent) => {
      if (isDragging && itemRef.current) {
        const clientX = 'changedTouches' in event ? event.changedTouches[0].clientX : event.clientX;
        const clientY = 'changedTouches' in event ? event.changedTouches[0].clientY : event.clientY;

        onDrop?.(itemRef.current, { x: clientX, y: clientY });
      }

      setIsDragging(false);
      setDragPosition({ x: 0, y: 0 });
      itemRef.current = null;
      startPosRef.current = null;

      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }

      onDragEnd?.();
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    document.addEventListener('touchmove', handleMouseMove, { passive: false });
    document.addEventListener('touchend', handleMouseUp);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.removeEventListener('touchmove', handleMouseMove);
      document.removeEventListener('touchend', handleMouseUp);
    };
  }, [isDragging, onDrag, onDragEnd, onDrop]);

  const transform = isDragging
    ? `translate(${dragPosition.x}px, ${dragPosition.y}px)`
    : undefined;

  return (
    <div
      onMouseDown={handleMouseDown}
      onTouchStart={handleMouseDown}
      style={{
        transform,
        cursor: disabled ? 'default' : isDragging ? 'grabbing' : 'grab',
        touchAction: 'none',
        userSelect: 'none',
      }}
    >
      {children}
    </div>
  );
};

// Node Draggable Wrapper
interface NodeDraggableProps {
  node: PipelineNode;
  onPositionChange: (id: string, position: NodePosition) => void;
  children: React.ReactNode;
}

export const NodeDraggable: React.FC<NodeDraggableProps> = ({
  node,
  onPositionChange,
  children,
}) => {
  const handleDrag = useCallback(
    (delta: NodePosition) => {
      onPositionChange(node.id, {
        x: node.position.x + delta.x,
        y: node.position.y + delta.y,
      });
    },
    [node.id, node.position, onPositionChange]
  );

  return (
    <Draggable onDrag={handleDrag}>
      {children}
    </Draggable>
  );
};

export default Draggable;
