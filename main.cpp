

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cstdio>

using namespace std;
using namespace cv;

#define ESC 27

int main( int argc, char** argv)
{
	CvCapture* capture;
	Mat frame, frame_gray;
	vector<Rect> faces;
	CascadeClassifier face_cascade;
	int c, i;
	int vid_dev = 0;/* default device */

	for(i = 1; i < argc ; ++i) {
		if(strcmp(argv[i], "-d") == 0) {
			vid_dev = atoi(argv[++i]);
		}
	}

	/* frontal face detection */
	if(!face_cascade.load("data/face_detect_frontal.xml")) {
		return -1;
	}

	capture = cvCaptureFromCAM(vid_dev);
	if(capture) {
		for(;;) {
			/* query a frame */
			frame = cvQueryFrame(capture);

			if(!frame.empty()) {
				/* gray conversion */
				cvtColor(frame, frame_gray, CV_BGR2GRAY);
				equalizeHist(frame_gray, frame_gray);
				face_cascade.detectMultiScale(frame_gray, faces,
						1.1, 3, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30));

				for(size_t i = 0; i < faces.size(); i++) {
					Point lefttop(faces[i].x,faces[i].y);
					Point rightbottom(faces[i].x + faces[i].width,
							faces[i].y + faces[i].height);
					/* Draw yellow color rectangle over the face */
					rectangle(frame, lefttop, rightbottom,
							Scalar(0, 255, 255), 1, 8, 0);
				}
				/* display */
				imshow("Demo : Face-detection", frame);
			}
			c = waitKey(10);
			/* Quit if ESC key is pressed */
			if(c == ESC) {
				break;
			}
		}
	}
	return 0;
}
