"""
模板下载器
提供模板下载、安装和版本管理功能
"""
import json
import os
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from .api import create_api, TemplateAPI


class TemplateDownloader:
    """模板下载器"""
    
    def __init__(self, templates_dir: str = None, install_dir: str = None):
        self.templates_dir = templates_dir or os.path.dirname(__file__)
        self.install_dir = install_dir or os.path.expanduser("~/templates")
        self.api = create_api(templates_dir)
        self.download_history = os.path.join(self.install_dir, ".download_history")
        
    def download_template(
        self,
        template_id: str,
        version: str = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """下载模板"""
        template = self.api.get_template(template_id)
        if not template:
            return {"success": False, "error": "Template not found"}
        
        template_dir = os.path.join(self.install_dir, template_id)
        
        if os.path.exists(template_dir) and not force:
            return {
                "success": False,
                "error": "Template already installed",
                "path": template_dir
            }
        
        # 创建目标目录
        os.makedirs(template_dir, exist_ok=True)
        
        # 复制文件
        files = self.api.get_template_files(template_id)
        for file_rel in files:
            src = os.path.join(self.templates_dir, file_rel)
            dst = os.path.join(template_dir, os.path.basename(file_rel))
            shutil.copy2(src, dst)
        
        # 保存元数据
        metadata = {
            **template,
            "installed_at": datetime.now().isoformat(),
            "local_path": template_dir
        }
        with open(os.path.join(template_dir, ".metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 记录下载历史
        self._record_download(template_id, metadata)
        
        return {
            "success": True,
            "template": template,
            "path": template_dir
        }
    
    def download_category(self, category_id: str) -> Dict[str, Any]:
        """下载整个分类"""
        category = self.api.get_category(category_id)
        if not category:
            return {"success": False, "error": "Category not found"}
        
        results = []
        for template in category["templates"]:
            result = self.download_template(template["id"])
            results.append(result)
        
        return {
            "success": all(r["success"] for r in results),
            "category": category,
            "results": results
        }
    
    def install_all_templates(self) -> Dict[str, Any]:
        """安装所有模板"""
        results = []
        for cat in self.api.index["categories"]:
            for template in cat["templates"]:
                result = self.download_template(template["id"])
                results.append(result)
        
        return {
            "success": all(r["success"] for r in results),
            "total": len(results),
            "downloaded": len([r for r in results if r["success"]]),
            "results": results
        }
    
    def list_installed(self) -> List[Dict]:
        """列出已安装的模板"""
        installed = []
        
        for item in os.listdir(self.install_dir):
            metadata_path = os.path.join(self.install_dir, item, ".metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                installed.append(metadata)
        
        return installed
    
    def remove_template(self, template_id: str) -> Dict[str, Any]:
        """移除已安装的模板"""
        template_dir = os.path.join(self.install_dir, template_id)
        
        if not os.path.exists(template_dir):
            return {"success": False, "error": "Template not installed"}
        
        shutil.rmtree(template_dir)
        
        return {"success": True, "template_id": template_id}
    
    def update_template(
        self,
        template_id: str,
        version: str = None
    ) -> Dict[str, Any]:
        """更新模板"""
        return self.download_template(template_id, version=version, force=True)
    
    def get_template_info(self, template_id: str) -> Optional[Dict]:
        """获取已安装模板信息"""
        metadata_path = os.path.join(
            self.install_dir, template_id, ".metadata.json"
        )
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None
    
    def _record_download(self, template_id: str, metadata: Dict):
        """记录下载历史"""
        history = []
        if os.path.exists(self.download_history):
            with open(self.download_history, 'r') as f:
                history = json.load(f)
        
        history.append({
            "template_id": template_id,
            "downloaded_at": datetime.now().isoformat(),
            "version": metadata.get("version")
        })
        
        with open(self.download_history, 'w') as f:
            json.dump(history, f, indent=2)
    
    def get_download_history(self, template_id: str = None) -> List[Dict]:
        """获取下载历史"""
        if not os.path.exists(self.download_history):
            return []
        
        with open(self.download_history, 'r') as f:
            history = json.load(f)
        
        if template_id:
            history = [h for h in history if h["template_id"] == template_id]
        
        return history
    
    def create_requirements_file(
        self,
        template_id: str,
        output_path: str = None
    ) -> str:
        """创建requirements.txt"""
        template = self.get_template_info(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not installed")
        
        requirements = template.get("requirements", [])
        content = "\n".join(requirements)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(content)
            return output_path
        
        return content


def create_downloader(
    templates_dir: str = None,
    install_dir: str = None
) -> TemplateDownloader:
    """创建下载器实例"""
    return TemplateDownloader(templates_dir, install_dir)


# CLI支持
def main():
    """CLI入口"""
    import sys
    
    downloader = create_downloader()
    
    if len(sys.argv) < 2:
        print("Usage: python downloader.py <command> [args]")
        print("Commands:")
        print("  list          - List installed templates")
        print("  install <id>  - Install a template")
        print("  install-all   - Install all templates")
        print("  remove <id>   - Remove a template")
        print("  update <id>   - Update a template")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "list":
        installed = downloader.list_installed()
        for t in installed:
            print(f"  {t['id']} - {t['name']} (v{t['version']})")
    
    elif command == "install":
        if len(sys.argv) < 3:
            print("Usage: python downloader.py install <template_id>")
            sys.exit(1)
        result = downloader.download_template(sys.argv[2])
        print(f"Success: {result.get('path', result.get('error'))}")
    
    elif command == "install-all":
        result = downloader.install_all_templates()
        print(f"Installed: {result['downloaded']}/{result['total']}")
    
    elif command == "remove":
        if len(sys.argv) < 3:
            print("Usage: python downloader.py remove <template_id>")
            sys.exit(1)
        result = downloader.remove_template(sys.argv[2])
        print(f"Removed: {result['success']}")
    
    elif command == "update":
        if len(sys.argv) < 3:
            print("Usage: python downloader.py update <template_id>")
            sys.exit(1)
        result = downloader.update_template(sys.argv[2])
        print(f"Updated: {result.get('path', result.get('error'))}")


if __name__ == "__main__":
    main()
