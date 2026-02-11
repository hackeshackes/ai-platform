with open('api/routes.py', 'r') as f:
    content = f.read()

# 注释掉v2.4 Plugin Marketplace
old_block = '''# v2.4: Plugin Marketplace
try:
    from api.endpoints import plugins
    router.include_router(plugins.router, prefix="/plugins", tags=["Plugins"])
    PLUGIN_MARKETPLACE_ENABLED = True
except ImportError:
    PLUGIN_MARKETPLACE_ENABLED = False'''

new_block = '''# v2.4: Plugin Marketplace (已禁用)
# try:
#     from api.endpoints import plugins
#     router.include_router(plugins.router, prefix="/plugins", tags=["Plugins"])
#     PLUGIN_MARKETPLACE_ENABLED = True
# except ImportError:
#     PLUGIN_MARKETPLACE_ENABLED = False'''

content = content.replace(old_block, new_block)

with open('api/routes.py', 'w') as f:
    f.write(content)

print("V2.4 plugins disabled")
