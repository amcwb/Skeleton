import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

class Skeleton:
    def __init__(self, boundaries):
        """
        Create a skeleton tracker, with colour boundaries.

        :param boundaries: The list of colour boundaries
        :type boundaries: List[Tuple[int, int, int]]
        """
        self.cap = None
        self.cam_ax = plt.subplot(2, 1, 1)
        self.vis_ax = plt.subplot(2, 1, 2)

        self._acquire_camera()
        self.cam_im = self.cam_ax.imshow(self._grab())
        self.vis_im = self.vis_ax.imshow(self._grab(), cmap='Reds')
        self._release_camera()
        
        self.bounds = [
            (np.array(lower, dtype="uint8"), np.array(upper, dtype="uint8"))
            for lower, upper in boundaries
        ]


    def _grab(self):
        """
        Get an RGB frame from the camera at time of execution

        :return: The frame
        :rtype: Any
        """
        _,frame = self.cap.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def _filter(self, frame):
        """
        Filter out data that is within the ranges given in __init__

        :param frame: The image frame
        :type frame: Any
        :return: The resultant filtered data
        :rtype: Any
        """
        # Get a sum of all the masks
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = sum(
            cv2.inRange(frame, lower, upper)
            for lower, upper in self.bounds
        )

        res = cv2.bitwise_and(frame, frame, mask=mask)
        return res
    
    def _find_rect(self, res):
        """
        Find a rect of the colour

        :param res: The result from the filter
        :type res: Any
        :return: The rects, or None if none are found
        :rtype: (int, int, int, int) or None
        """
        # CREDIT: https://stackoverflow.com/a/31402351
        rows = np.any(res, axis=1)
        cols = np.any(res, axis=0)
        try:
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
        except IndexError:
            # None found
            return None
        else:
            return rmin, rmax, cmin, cmax

    def _draw_rect(self, rect, ax):
        # Draw a rect in the ax of mpl
        box = Rectangle(
            (rect[2], rect[0]),
            rect[3] - rect[2],
            rect[1] - rect[0],
            fill=False,
            facecolor='none',
            color='red',
            alpha=1,
        )

        ax.add_patch(box)

        return box

    def run(self):
        plt.ion()

        # Acquire the camera for the duration of the example
        self._acquire_camera()
        box_cam, box_vis = None, None
        while True:
            try:
                # Get the data
                data = self._grab()
                # Filter out the colours we were given in __init__
                res = self._filter(data)
                # Create a rect
                rect = self._find_rect(res)
                
                # Show the original colour image
                self.cam_im.set_data(data)
                # Only show detected colours
                self.vis_im.set_data(cv2.cvtColor(res, cv2.COLOR_HSV2RGB))

                if box_cam:
                    try:
                        box_cam.remove()
                    except ValueError:
                        pass
                if box_vis:
                    try:
                        box_vis.remove()
                    except ValueError:
                        pass
                
                if rect is not None:
                    # Draw colour rect on cam and visualisation
                    box_cam = self._draw_rect(rect, self.cam_ax)
                    box_vis = self._draw_rect(rect, self.vis_ax)
                
                # Prevent overworking
                plt.pause(1/60)
            except KeyboardInterrupt:
                plt.close()
                break

        self._release_camera()

    def _acquire_camera(self):
        # Acquire camera
        self.cap = cv2.VideoCapture(0)
    
    def _release_camera(self):
        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None