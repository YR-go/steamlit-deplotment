# app2.py
import foo
import bar
from multiapp import MultiApp
app = MultiApp()
app.add_app("Foo", foo.app)
app.add_app("Bar", bar.app)
app.run()