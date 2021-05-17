import importlib
from flask import Flask, request
from flask_restful import Resource, Api
from gunicorn.app.base import BaseApplication
import conf

app = Flask(__name__)
api = Api(app)


class PingService(Resource):
    def get(self):
        return 'OK', 200, {'Content-Type': 'text/plain'}

    def post(self):
        pass
        
class StandaloneApplication(BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super(StandaloneApplication, self).__init__()

    def load_config(self):
        config = dict([(key, value) for key, value in self.options.items()
                       if key in self.cfg.settings and value is not None])
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        api.add_resource(
            PingService,
            "/ping"
        )

        return self.application


if __name__ == '__main__':
    options = {'bind': '%s:%s' % ('127.0.0.1', '8080'),}
    application = StandaloneApplication(app, options).run()