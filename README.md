# OpenCV for LabVIEW
OpenCV is the de facto industry standard for open-sourced, computer vision. Written in C++ and wrapped up for Python, OpenCV can now be implemented in LabVIEW through the Python Node.

# SubVIs to add
<ul>
  <li>General convolution<br />
    cv2.filter2D()<br />
    https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/</li>
  <li>Colorspace<br />
    cv2.cvtColor()</li>
  <li>Color channels<br />
    cv2.split()</li>
  <li>Blur<br />
    cv2.blur()</li>
  <li>Contours<br />
    cv2.findContours()</li>
  <li>Harris corner<br />
    cv2.cornerHarris()</li>
  <li>Save frames<br />
    cv2.imwrite()</li>
  <li>Translation<br />
    cv2.warpAffine()</li>
  <li>Rotation<br />
    cv2.warpAffine()</li>
  <li>Histograms<br />
    cv2.calcHist()</li>
  <li>Drawing<br />
    cv2.rectangle()/circle()/[shape]()</li>
  <li>Color detection<br />
    cv2.inRange()</li>
  <li>Decoding<br />
    cv2.QRCodeDetector.[]()</li>
  <li>CNN/DeepNN<br />
    cv2.dnn()</li>
  <li>Resize/zoom **<br />
    cv2.resize()</li>
  <li>Crop **<br />
    numpy indexing</li>
  <li>ROI/Masks **<br />
    cv2.bitwise_and(mask=kwarg)</li>
  <br />
  ** Resizes image (would require undoing hardcoding of 640x480)
</ul>

# SubVIs included
<ul>
  <li>Initialize CV Session<br />
    cv2.VideoCapture()</li>
  <li>Close CV Session<br />
    self.caprelease(), cv2.destroyAllWindows()</li>
  <li>Snap Image<br />
    self.cap.read()</li>
  <li>Python Debug<br />
    Python print() to LabVIEW front panel</li>
  <li>Python Path<br />
    path to python source file</li>
  <li>Flip Horizontal/Vetical<br />
    cv2.flip()</li>
  <li>Face Detection/Tracking<br />
    cv2.cascadeClassifier()</li>
  <li>Edge Detection<br />
    cv2.Canny()</li>
 </ul>
