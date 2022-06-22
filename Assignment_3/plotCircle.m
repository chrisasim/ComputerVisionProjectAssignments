function [] = plotCircle(radius,x,y)
th = 0:pi/50:2*pi;
xunit = radius * cos(th) + x;
yunit = radius * sin(th) + y;
h = plot(xunit, yunit);