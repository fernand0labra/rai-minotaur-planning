from visualization.plotter import Plotter

class Animator:
    def __init__(self,environment):
        self.env=environment

    def animate(
        self,
        path=None,
        extra=None,
        start=None,
        goal=None,
        title="Planner"
    ):

        if self.env.is3d:
            print("3D animation disabled. Showing final result.")
            Plotter(self.env).draw(
                path,
                extra,
                start,
                goal,
                title
            )

        else:
            Plotter(self.env).draw(
                path,
                extra,
                start,
                goal,
                title
            )