import React, { useState, useRef, useEffect, useCallback } from 'react';
import './styles.css';

interface CodeEditorProps {
  initialCode?: string;
  language?: string;
  theme?: 'vs-dark' | 'vs-light' | 'hc-black';
  readOnly?: boolean;
  showLineNumbers?: boolean;
  highlightLines?: number[];
  onChange?: (code: string) => void;
  onRun?: (code: string) => void;
  onCopy?: (code: string) => void;
  placeholder?: string;
  autoResize?: boolean;
  maxHeight?: string;
}

export const CodeEditor: React.FC<CodeEditorProps> = ({
  initialCode = '',
  language = 'javascript',
  theme = 'vs-dark',
  readOnly = false,
  showLineNumbers = true,
  highlightLines = [],
  onChange,
  onRun,
  onCopy,
  placeholder = '// 在这里输入代码...',
  autoResize = false,
  maxHeight = '400px'
}) => {
  const [code, setCode] = useState(initialCode);
  const [isEditing, setIsEditing] = useState(false);
  const [cursorPosition, setCursorPosition] = useState({ line: 1, column: 1 });
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (initialCode !== code) {
      setCode(initialCode);
    }
  }, [initialCode]);

  const lines = code.split('\n');
  const lineCount = Math.max(lines.length, 1);

  const handleCodeChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newCode = e.target.value;
    setCode(newCode);
    
    const textBeforeCursor = newCode.substring(0, e.target.selectionStart);
    const linesBeforeCursor = textBeforeCursor.split('\n');
    setCursorPosition({
      line: linesBeforeCursor.length,
      column: linesBeforeCursor[linesBeforeCursor.length - 1].length + 1
    });

    if (onChange) {
      onChange(newCode);
    }
  }, [onChange]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Tab') {
      e.preventDefault();
      const textarea = textareaRef.current;
      if (!textarea) return;

      const start = textarea.selectionStart;
      const end = textarea.selectionEnd;
      const newCode = code.substring(0, start) + '  ' + code.substring(end);
      setCode(newCode);
      
      setTimeout(() => {
        textarea.selectionStart = textarea.selectionEnd = start + 2;
      }, 0);

      if (onChange) {
        onChange(newCode);
      }
    }

    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && onRun) {
      e.preventDefault();
      onRun(code);
    }
  }, [code, onChange, onRun]);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(code);
      if (onCopy) {
        onCopy(code);
      }
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  }, [code, onCopy]);

  const handleRun = useCallback(() => {
    if (onRun) {
      onRun(code);
    }
  }, [code, onRun]);

  const handleReset = useCallback(() => {
    setCode(initialCode);
    if (onChange) {
      onChange(initialCode);
    }
  }, [initialCode, onChange]);

  const getLanguageLabel = (lang: string): string => {
    const labels: Record<string, string> = {
      javascript: 'JavaScript',
      typescript: 'TypeScript',
      python: 'Python',
      java: 'Java',
      cpp: 'C++',
      go: 'Go',
      rust: 'Rust',
      ruby: 'Ruby',
      php: 'PHP',
      swift: 'Swift',
      kotlin: 'Kotlin',
      html: 'HTML',
      css: 'CSS',
      sql: 'SQL',
      bash: 'Bash',
      json: 'JSON',
      yaml: 'YAML'
    };
    return labels[lang] || lang.toUpperCase();
  };

  return (
    <div 
      ref={containerRef}
      className={`code-editor-container ${theme} ${readOnly ? 'readonly' : ''} ${isEditing ? 'editing' : ''}`}
      style={{ maxHeight }}
    >
      <div className="editor-header">
        <div className="header-left">
          <span className="language-badge">{getLanguageLabel(language)}</span>
        </div>
        <div className="header-right">
          {!readOnly && (
            <>
              <button 
                className="header-btn" 
                onClick={handleReset}
                title="重置代码"
              >
                🔄 重置
              </button>
              <button 
                className="header-btn" 
                onClick={handleCopy}
                title="复制代码"
              >
                📋 复制
              </button>
              {onRun && (
                <button 
                  className="header-btn run-btn" 
                  onClick={handleRun}
                  title="运行代码 (Ctrl+Enter)"
                >
                  ▶️ 运行
                </button>
              )}
            </>
          )}
        </div>
      </div>

      <div className="editor-body">
        {showLineNumbers && (
          <div className="line-numbers">
            {Array.from({ length: lineCount }, (_, i) => (
              <div 
                key={i} 
                className={`line-number ${highlightLines.includes(i + 1) ? 'highlighted' : ''}`}
              >
                {i + 1}
              </div>
            ))}
          </div>
        )}
        
        <div className="code-area-wrapper">
          {/* 高亮层 */}
          <pre className="code-highlight" aria-hidden="true">
            <code dangerouslySetInnerHTML={{ 
              __html: code
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/(\/\/.*$)/gm, '<span class="comment">$1</span>')
                .replace(/(\/\*[\s\S]*?\*\/)/g, '<span class="comment">$1</span>')
                .replace(/\b(const|let|var|function|return|if|else|for|while|class|extends|import|export|from|async|await|try|catch|throw|new|this|super|static|public|private|protected|interface|enum|type)\b/g, '<span class="keyword">$1</span>')
                .replace(/\b(true|false|null|undefined|NaN|Infinity)\b/g, '<span class="boolean">$1</span>')
                .replace(/\b(\d+\.?\d*)\b/g, '<span class="number">$1</span>')
                .replace(/(["'`])([^"'`]*)\1/g, '<span class="string">$1$2$1</span>')
            }} />
          </pre>
          
          {/* 编辑区 */}
          <textarea
            ref={textareaRef}
            className="code-textarea"
            value={code}
            onChange={handleCodeChange}
            onKeyDown={handleKeyDown}
            onFocus={() => setIsEditing(true)}
            onBlur={() => setIsEditing(false)}
            readOnly={readOnly}
            placeholder={placeholder}
            spellCheck={false}
            autoCapitalize="off"
            autoCorrect="off"
            autoComplete="off"
          />
        </div>
      </div>

      <div className="editor-footer">
        <div className="footer-left">
          <span className="cursor-position">
            行 {cursorPosition.line}, 列 {cursorPosition.column}
          </span>
          <span className="line-count">
            {lineCount} 行
          </span>
        </div>
        <div className="footer-right">
          {onRun && (
            <span className="shortcut-hint">
              Ctrl+Enter 运行
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

export default CodeEditor;
