import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"API Flask na porta {port}...")
    print(f"http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)

