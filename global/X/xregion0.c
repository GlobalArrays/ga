/*$Id: xregion0.c,v 1.2 1995-02-02 23:12:59 d3g681 Exp $*/
/*
 * A visualization program for GAs
 *
 * Jarek Nieplocha, 11.02.1993
 *
 */ 

#include <sys/types.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
 
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/StringDefs.h>
#include <X11/Intrinsic.h>
#include <X11/IntrinsicP.h>
#include <X11/Shell.h>
#include <X11/ShellP.h>
#include <X11/Xaw/Scrollbar.h>
#include <X11/Xaw/Form.h>
#include <X11/Xaw/Command.h>
#include <X11/Xaw/Label.h>

#define DEBUG 0
#define INITSLOW 50L

extern char *malloc();
extern void exit();

/* Globals needed for display etc. */

static  Widget top_level_widget, box_widget, start_stop_button,
               scroll_widget, interval_widget,
               scroll_widget2, slowdown_widget,
               map_widget,
               quit_button, canvas_widget, title_widget;
static XtAppContext app_context;
static  XtIntervalId timer;
static  long first_time = True; /* Used to set scroll bar on first expose */

static  long interval_max = 2000;
static  long interval = 500;         /* 0.5s between exposures by default */

static  long slowdown_max = 100;
static  long slowdown = INITSLOW;        /* slowdown factor for animation */
static  long oldslowdown = INITSLOW;

static  unsigned long int cur_time=0; /* current time */

static  Dimension frame_size;
static  Arg arg[25]; 
static  Display *display;
static  Window window, window_map;
static  int screen, depth;
static  Visual *visual;
static  XImage *image;
static  u_char *pict;
static  GC gc, gc_map;
static  char title[80];
static  char interval_string[10], slowdown_string[10];

#define MAX(a,b) (((a)>(b)) ? (a) : (b))
#define MIN(a,b) (((a)<(b)) ? (a) : (b))

#define MAX_COL 16
static u_char cmap[MAX_COL+1];
Colormap colormap;

static int grid_size;		/* The size of the grid */
static int scale;		/* No. of pixels per element */
static int pict_size;		/* The size of the piture = grid_size*scale */

static  u_char *grid;


static  double *ltime;          /* last event time */
static  double *integr;         /* access integral */
static  double maxval=0.;       /* max value of integral, zero is default */
static  u_char *flag;           /* access flag */


static  int working = False, animation = True;

/*** trace variables and constants ***/

static  long int num_events;
static  int *record;			/* tracefile data */
static  unsigned long int *times;	/* times of events */
static  int cur_event=0;
#define RECLEN 8

/*** end of trace variables and constants ***/


 
void Error(string, integer)
     char *string;
     int integer;
{
  (void) fflush(stdout);
  (void) fprintf(stderr,"\n\nError was called.\n");
  (void) fprintf(stderr,string);
  (void) fprintf(stderr," %d (%#x).\n",integer,integer);
  exit(1);
}


void UpdatePixRegion(ilo, ihi, jlo, jhi, increment, time)
     int ilo, ihi, jlo, jhi, increment;
     double time;
{
  register int i, j, k, l, index;
  register u_char *from, *to, *tempk, *tempkl, value, *pflag;
  register double *pintegr, *pltime, corr;
  

  for (i=ilo; i<=ihi; i++)
    for (j=jlo; j<=jhi; j++) {

      to   = pict + (i*pict_size + j)*scale;
      from = grid + (i*grid_size + j);
      pflag = flag + (i*grid_size + j);
      pintegr = integr + (i*grid_size + j);
      pltime  = ltime  + (i*grid_size + j);

      /* increment == 0 means animation is done and displaying integrals */

      if(animation){
          corr      = *from >1 ? *from -1 : 0;
          corr      = corr  * (time - *pltime);

          if(corr<0.){
           printf("error: time =%f ltime =%f height =%d \n",time,*pltime,*from);
          }

          *pintegr += corr;

          *pltime   = time;

          *from = *from + increment;
          index = *from;
          index = MIN(index,MAX_COL-1);

          if(increment)*pflag = 1;

          /*  calculate max value of integrals */
          if(*pintegr > maxval)
                 maxval = *pintegr;

      }else{
 
          index = (int)(((*pintegr)/maxval)* MAX_COL);
          index = MIN(index,MAX_COL-1);
          if(!index && *pflag) index = MAX_COL;  /* sets the "accessed" color */
      }
          
      
      value     = cmap[index];

      for (k=0, tempk=to; k<scale; k++, tempk+=pict_size)
	for (l=0, tempkl=tempk; l<scale; l++, tempkl++)
	  *tempkl = value;
    }
}


double IntegralSum( ilo, ihi, jlo, jhi)
     int ilo, ihi, jlo, jhi;
{
/* Sum integrals for the specified part of the matrix */

  register int i, j;
  register double *pintegr, sum;

  sum = 0.0;
  for (i=ilo; i<=ihi; i++)
    for (j=jlo; j<=jhi; j++) {

      pintegr = integr + (i*grid_size + j);
      sum += *pintegr;
    }
  return(sum);
}


void DisplayPixRegion(ilo, ihi, jlo, jhi)
     int ilo, ihi, jlo, jhi;
{
  int y = ilo*scale;
  int x = jlo*scale;
  int height= (ihi-ilo+1)*scale;
  int width= (jhi-jlo+1)*scale;

  XPutImage(display, window, gc, image, x, y, x, y, width, height);
  XFlush(display);
}

void FileInputActions()
{
  int ilo, ihi, jlo, jhi, inc;

  if (scanf("%d %d %d %d %d", &ilo, &ihi, &jlo, &jhi, &inc) != 5)
    Error("FileInputActions: invalid input", -1);

  UpdatePixRegion(ilo, ihi, jlo, jhi, inc);
  DisplayPixRegion(ilo, ihi, jlo, jhi);
}
  
/**/
void FileInput(clientData, source, inputid)
     caddr_t clientData;
     int *source;
     XtInputId *inputid;
{
  static int toggle=0;
  
  /* There seem to be two events generated each time we should
     read the pipe .... ignore the second */
  
  if (toggle) {
    toggle = 0;
    return;
  }
  
  toggle = 1;
  
  FileInputActions();
}
  
void DisplayIntervalValue()
{
  (void) sprintf(interval_string, "%4d ms", interval);
  XtSetArg(arg[0], XtNlabel, interval_string);
  XtSetValues(interval_widget,arg,1);
}

void DisplaySlowdownValue()
{
  (void) sprintf(slowdown_string, "%4d times", slowdown);
  XtSetArg(arg[0], XtNlabel, slowdown_string);
  XtSetValues(slowdown_widget,arg,1);
}

/**/
void ScrollProc(scrollbar, data, position)
     Widget scrollbar;
     caddr_t data;
     caddr_t position;
/*
  Called when the left or right buttons are used to step the
  scrollbar left or right. We have the responsibility of
  moving the scrollbar.
*/
{
  Dimension length;
  float fraction, shown;

  /* Get the scrollbar length and move the scroll bar */

  XtSetArg(arg[0], XtNlength, &length);
  XtGetValues(scrollbar, arg, 1);
  fraction = ((int) position)/ (double) length;

  interval -= fraction*0.05*interval_max;   
  interval = MIN(interval, interval_max);
  interval = MAX(interval, 1);

  fraction = (float) interval/ (float) interval_max;
  shown = -1.0;

  DisplayIntervalValue();
  XawScrollbarSetThumb(scrollbar, fraction, shown);
}

/***** slowdown **********/
void ScrollProc2(scrollbar, data, position)
     Widget scrollbar;
     caddr_t data;
     caddr_t position;
/*
  Called when the left or right buttons are used to step the
  scrollbar left or right. We have the responsibility of
  moving the scrollbar.
*/
{
  Dimension length;
  float fraction, shown;

  /* Get the scrollbar length and move the scroll bar */

  XtSetArg(arg[0], XtNlength, &length);
  XtGetValues(scrollbar, arg, 1);
  fraction = ((int) position)/ (double) length;

  slowdown -= fraction*0.05*slowdown_max;   
  slowdown = MIN(slowdown,slowdown_max);
  slowdown = MAX(slowdown, 1);

  /* scale current time according to the slowdown factor */
  if(DEBUG) printf("before scaling %lu ( %ld %ld factor=%f) ",cur_time,
            slowdown,oldslowdown,(1.0*slowdown)/oldslowdown);

  cur_time = cur_time*slowdown/oldslowdown;

  if(DEBUG)printf("and after %lu\n ",cur_time);
  oldslowdown = slowdown;

  fraction = (float) slowdown/ (float) slowdown_max;
  shown = -1.0;

  DisplaySlowdownValue();
  XawScrollbarSetThumb(scrollbar, fraction, shown);
}
 

/**/
void JumpProc(scrollbar, data, fraction_ptr)
     Widget scrollbar;
     caddr_t data;
     caddr_t fraction_ptr;
/*
  Called when the middle button is used to drag to 
  the scrollbar. The scrollbar is moved for us.
*/
{
  float fraction = *(float *) fraction_ptr;

  interval = fraction*interval_max;
  interval = MIN(interval, interval_max);
  interval = MAX(interval, 1);

  DisplayIntervalValue();
}

/**** slowdown ****/
void JumpProc2(scrollbar, data, fraction_ptr)
     Widget scrollbar;
     caddr_t data;
     caddr_t fraction_ptr;
/*
  Called when the middle button is used to drag to 
  the scrollbar. The scrollbar is moved for us.
*/
{
  float fraction = *(float *) fraction_ptr;

  slowdown = fraction*slowdown_max;
  slowdown = MIN(slowdown, slowdown_max);
  slowdown = MAX(slowdown, 1);

  /* scale current time according to the slowdown factor */
  if(DEBUG) printf("before scaling %lu ( %ld %ld factor=%f) ",cur_time,
            slowdown,oldslowdown,(1.0*slowdown)/oldslowdown);

  cur_time = cur_time*slowdown/oldslowdown;

  if(DEBUG)printf("and after %lu\n ",cur_time);
  oldslowdown = slowdown;

  DisplaySlowdownValue();
}


void DrawColorMap()
{
/* Actually fill in the colors */
int i, black=1;
unsigned width= 20, height = pict_size/MAX_COL;
int x, y, length,index;
XColor color;

    for (i=0,y=0,x=0;i<MAX_COL;i++){
        index = (animation==False && i==0)? MAX_COL : i; 
        XSetForeground(display, gc_map, (unsigned long) cmap[index]);
        XFillRectangle(display,window_map,gc_map,x,y,width,height);
        y += height;
       /* XSetForeground(display, gc_map, (unsigned long) cmap[index]);*/
    }
}



void PrintColorMapText()
{
/* Print Legend numbers */
int i, black=1;
unsigned width= 20, height = pict_size/MAX_COL;
int x, y, length;
char string[9];
XColor color;
unsigned long textcolor;
double factor, intval;
int formatE;

/* use black or white for text color */

    if(black){ color.red   = 0; color.blue  = 0; color.green = 0;}
      else   { color.red   = 65535; color.blue  = 65535; color.green = 65535;}

    if (XAllocColor(display, colormap, &color) == 0)
                   Error("couldn't assign color",99);
    textcolor = (unsigned long)color.pixel;
    XSetForeground(display, gc_map, textcolor);

    x = width+4;
    y = 0;
    factor  = maxval/(MAX_COL);
    formatE = (maxval >= 10. || maxval < .001) ? 1 : 0; 
    for (i=0;i<MAX_COL;i++){
        y += height;
        if(animation)
           sprintf(string,"%2d",i);
        else{
           intval = factor*i;
           if(formatE) sprintf(string,"%5.1e",intval); 
           else        sprintf(string,"%6.4f",intval);
        }
        length = strlen(string);
        XDrawString(display,window_map,gc_map,x,y-2,string, length);
    }
}



void UpdateDisplay()
{
int n;
char string[60];
XFontStruct *labelfont;
char *fontname = "8x13bold";
Window W;

  /* Incorporate the display changes after the animation is over */


  /* change the title */

  strcpy(title, "Time Lost Due To Contention");
  XtSetArg(arg[0], XtNlabel, title); 
  XtSetValues(title_widget,arg,1);


  /* delete all irrelevant widgets */

  XtDestroyWidget(interval_widget); 
  XtDestroyWidget(scroll_widget);
  XtDestroyWidget(slowdown_widget);
  XtDestroyWidget(scroll_widget2);
  XtDestroyWidget(start_stop_button);
  
  /* load the label font */
  
  if ((labelfont = XLoadQueryFont(display,fontname)) == NULL)
     Error("failed to load label font",-1);
       
  
  /* Add label widget  to display the max value of integrals */

  n = 0;
  (void) sprintf(string, "max value = %8.5f", maxval);
  XtSetArg(arg[n], XtNx, 10); n++;
  XtSetArg(arg[n], XtNwidth, frame_size); n++;
  XtSetArg(arg[n], XtNvertDistance, 2); n++;
  XtSetArg(arg[n], XtNfromVert, title_widget); n++;
  /*  XtSetArg(arg[n], XtNhorizDistance,90 ); n++;
  XtSetArg(arg[n], XtNfromHoriz, quit_button); n++;*/
  /* XtSetArg(arg[n], XtNfont, labelfont); n++; this doesn't work here, why ?*/
  XtSetArg(arg[n], XtNlabel, string); n++;
  XtSetArg(arg[n], XtNborderWidth, (Dimension) 0); n++;
  interval_widget = XtCreateManagedWidget("maxval", labelWidgetClass,
                                           box_widget, arg, n);


  XtSetArg(arg[0], XtNfont, labelfont);  /* but it does here */
  XtSetValues(interval_widget,arg,1);

  /* Raise Quit widget partially obscurred by centered max value string */

  W = XtWindow(quit_button);
  XRaiseWindow(display,W);
 
  /* Now, update the ColorMap legend */

  XClearWindow(display, window_map);
  DrawColorMap();
  PrintColorMapText();

  XFlush(display);
}


extern long random();

/*****************************************************************/
void TimeOutCallback(data)
     caddr_t data;
{
#define  AMP 1
int ilo, ihi, jlo, jhi, inc;
int base, stime;


/* Do work on time out here */

/*
     printf("time %lu, event=%d\n",cur_time,cur_event);
     fflush(stdout);
     printf("times[cur_event]= ");
*/
     while( slowdown*(times[cur_event]/1000) < (cur_time + interval) &&
						 cur_event <num_events){
        base = cur_event*RECLEN;
        ilo = record[base+2];
        ihi = record[base+3];
        jlo = record[base+4];
        jhi = record[base+5];
    
        inc = AMP*record[base+7];
        UpdatePixRegion(ilo, ihi, jlo, jhi, inc, (double)1e-6*times[cur_event]);
        DisplayPixRegion(ilo, ihi, jlo, jhi);
     
        cur_event++;
    } 
    if(cur_event <num_events){
        cur_time += interval;
    }
    else if(animation){

        animation = False;
        printf("\nEnd of Event Animation ...\n");
        UpdatePixRegion(0, grid_size-1, 0, grid_size-1,  0, 0.0);
        DisplayPixRegion( 0, grid_size-1, 0, grid_size-1);
        UpdateDisplay();

        if(DEBUG){
            for(inc=0;inc<grid_size*grid_size;inc++)printf("%f\n",integr[inc]);
            printf("Intialized or computed max integral value = %f\n",maxval); 
        }
    }  

    /* Restore the call back  */
    timer = XtAddTimeOut(interval, TimeOutCallback, NULL);
}  


/**/
void Exposed(widget, data, event)
     Widget widget;
     caddr_t data;
     XEvent *event;
{
  /* Now we are exposed so we can draw ... */

  if (first_time) {
    /* Cannot seem to set this before now ? */
    ScrollProc(scroll_widget, NULL, 0);
    ScrollProc2(scroll_widget2, NULL, 0);
    first_time = False;
  }

  DrawColorMap();
  PrintColorMapText();
  DisplayPixRegion(0, grid_size-1, 0, grid_size-1);
  XFlush(display);
}  


/**/
void Quit(widget, data, event)
     Widget widget;
     caddr_t data;
     XEvent *event;
{
  exit(0);
}


/**/
void StartStop(widget, data, event)
     Widget widget;
     caddr_t data;
     XEvent *event;
{
  /* Toggle propagation of display */

  if (working) {
    XtRemoveTimeOut(timer);
    working = False;
    XtSetArg(arg[0], XtNlabel, "Start");   /* Reset button label */
    XtSetValues(start_stop_button,arg,1);
    XFlush(display);
  }
  else {
    XtSetArg(arg[0], XtNlabel, "Stop");   /* Reset button label */
    XtSetValues(start_stop_button,arg,1);
    timer = XtAddTimeOut(interval, TimeOutCallback, NULL);
    working = True;
    XFlush(display);
  }
}  

void HSVtoRGB(h, s, v, r, g, b)
     double h, s, v, *r, *g, *b;
/*
  hue (0-360), s&v (0-1), r&g&b (0-1)
*/
{
  int ih;
  double rh, x, y, z;

  /* Zero saturation means gray */

  if (s < 0.0001) {
    *r = v; *g = v; *b = v;
    return;
  }

  /* Put hue in [0,6) */

  if (h > 359.999)
    h = 0.0;
  else
    h = h/60.0;

  ih = h; rh = h - ih;     /* Integer and fractional parts */

  x = v*(1.0 - s);
  y = v*(1.0-s*rh);
  z = v*(1.0-s*(1.0-rh));

  switch (ih) {
  case 0:
    *r = v; *g = z; *b = x; break;
  case 1:
    *r = y; *g = v; *b = x; break;
  case 2:
    *r = x; *g = v; *b = z; break;
  case 3:
    *r = x; *g = y; *b = v; break;
  case 4:
    *r = z; *g = x; *b = v; break;
  case 5:
    *r = v; *g = x; *b = y; break;
  default:
    Error("HLStoRGB: invalid hue", ih);
  }
}

void Setcmap()
/*
  Make the color map ... 
*/
{
  int i;
  XColor color;
  double cscale = 1.0 / ((double) (MAX_COL-1));
  double hue, saturation, value;
  double redvar,red=0xba/255.0, green, blue=0xd2/255.0;
  redvar = red;

  colormap = DefaultColormap(display, screen);

  /* Linear interpolation on green */

  for (i=0; i<MAX_COL; i++) {
    if (i == 0) {
      /* Assign white as the first color */
      color.red   = 65535;
      color.blue  = 65535;
      color.green = 65535;
    }
    else {
      if(i>=(MAX_COL-5)) redvar = .4*red + .6*red*(MAX_COL-i)/5.; 
      color.red   = (short) (redvar   * 65535.0);
      color.blue  = (short) (blue  * 65535.0);
      green = cscale * (MAX_COL - i);
      color.green = (short) (green * 65535.0);
    }

    if (XAllocColor(display, colormap, &color) == 0)
      Error("couldn't assign color",i);

    cmap[i] = color.pixel;
 
/* now set the "accessed" color for regions that were accessed */
    color.red   = 65535;
    color.green = 65535;
    color.blue  = 65535 * (220.0/255.0);
    if (XAllocColor(display, colormap, &color) == 0)
      Error("couldn't assign accessed color",99);
    cmap[MAX_COL] = color.pixel; 

/*
    (void) printf("Colour %d red=%x, green=%x, blue=%x, pixel=%x\n",
		  i, color.red, color.green, color.blue, color.pixel);
*/
  }

  /* Unused */
  /* Get rgb color from ascii name */
  /*    if (XParseColor(display, colormap, colors[i], &color) == 0)
	Error("Setcmap: error parsing color number ",i); */

}




void ReadEventFile(filename)
     char *filename;
{
  FILE *fin;
  long int i,k,act_events=0;
  
  fin = fopen(filename,"r");
  if(!fin) Error("Input File Not Found",-2);
  if(!(record = (int*)malloc(RECLEN*num_events*sizeof(int))))
    Error("couldn't allocate memory",-1);
  if(!(times = (unsigned long int *)malloc(num_events*sizeof(unsigned long))))
    Error("couldn't allocate memory",-2);

  for(i=0;i<num_events;i++){
    for(k=0;k<RECLEN;k++) fscanf(fin,"%d",(i*RECLEN+k)+record);
    if(fscanf(fin,"%lu",times+i))act_events++;
    
    /* Adjust from Fortran to C base addressing */
    
    for (k=2; k<=5; k++)
      (*((i*RECLEN+k)+record))--;
    if(feof(fin))break;
  }
  num_events = act_events;
 printf("File %s has been read. %d events are displayed\n",filename,num_events);

  fclose(fin);
}



int main(argc, argv)
     unsigned int argc;
     char **argv;
{
  int n,i, argn;
  char filename[80];
  Font LegendFont;
  XGCValues gcv;

  /* First read the argument list */ 
  for(i=1,argn=0;i<argc;i++){
     if(argv[i][0] == '-') break;
     else argn =i; 
  }
  argn ++;

  if(argn <5 ){
    printf("Usage:\n");
    printf("xregion <grid_size> <scale> <filename> <num_events> [max_integr]\
    [Xtoolkit options]\n ");
    exit(1);
  }
  sscanf(argv[1],"%d", &grid_size);
  sscanf(argv[2],"%d", &scale);
  sscanf(argv[3],"%s", filename);
  sscanf(argv[4],"%d", &num_events);
  if(argn>5)sscanf(argv[6],"%d",&maxval);
  
  ReadEventFile(filename);

  pict_size = grid_size * scale;

  frame_size = MAX(pict_size, 512);

  /* Create top level shell widget */
  
  top_level_widget = XtInitialize("toplevel", "TopLevel",
				  NULL, 0, &argc, argv);

  /* Create form widget to hold everything else */

  n = 0;
  box_widget = XtCreateManagedWidget("box", formWidgetClass, 
				     top_level_widget, arg, n);

  /* Create the label to hold the title */

  (void) strcpy(title, "Array Access Display");
  n = 0;
  XtSetArg(arg[n], XtNx, 10); n++;
  XtSetArg(arg[n], XtNy, 10); n++;
  XtSetArg(arg[n], XtNwidth, frame_size); n++;
  XtSetArg(arg[n], XtNlabel, title); n++;
  XtSetArg(arg[n], XtNborderWidth, (Dimension) 0); n++;
  title_widget = XtCreateManagedWidget("title", labelWidgetClass,
				       box_widget, arg, n);

  /* Create the Quit command button */

  n = 0;
  XtSetArg(arg[n], XtNx, 10); n++;
  XtSetArg(arg[n], XtNvertDistance, 10); n++;
  XtSetArg(arg[n], XtNfromVert, title_widget); n++;
  XtSetArg(arg[n], XtNlabel, "Quit"); n++;
  XtSetArg(arg[n], XtNshapeStyle, XmuShapeOval); n++;
  quit_button = XtCreateManagedWidget("quit", commandWidgetClass,
				       box_widget, arg, n);
  XtAddCallback(quit_button, XtNcallback, Quit, NULL);

  /* Create the Start/Stop command button */

  n = 0;
  XtSetArg(arg[n], XtNvertDistance, 10); n++;
  XtSetArg(arg[n], XtNfromVert, title_widget); n++;
  XtSetArg(arg[n], XtNhorizDistance, 10); n++;
  XtSetArg(arg[n], XtNfromHoriz, quit_button); n++;
  XtSetArg(arg[n], XtNlabel, "Start"); n++;
  XtSetArg(arg[n], XtNshapeStyle, XmuShapeOval); n++;
  start_stop_button = XtCreateManagedWidget("start/stop", commandWidgetClass,
				       box_widget, arg, n);
  XtAddCallback(start_stop_button, XtNcallback, StartStop, NULL);

  /* Create the scroll bar for the interval */

  n = 0;
  XtSetArg(arg[n], XtNvertDistance, 15); n++;
  XtSetArg(arg[n], XtNfromVert, title_widget); n++;
  XtSetArg(arg[n], XtNhorizDistance, 20); n++;
  XtSetArg(arg[n], XtNfromHoriz, start_stop_button); n++;
  XtSetArg(arg[n], XtNorientation, XtorientHorizontal); n++;
  XtSetArg(arg[n], XtNlength, 100); n++;
  XtSetArg(arg[n], XtNthickness, 15); n++;
  scroll_widget = XtCreateManagedWidget("scroll", scrollbarWidgetClass,
					box_widget, arg, n);
  XtAddCallback(scroll_widget, XtNscrollProc, ScrollProc, NULL);
  XtAddCallback(scroll_widget, XtNjumpProc, JumpProc, NULL);

  /* Create the label widget which displays the interval value
     associated with the scrollbar. */

  n = 0;
  (void) sprintf(interval_string, "%4d ms", interval);
  XtSetArg(arg[n], XtNvertDistance, 10); n++;
  XtSetArg(arg[n], XtNfromVert, title_widget); n++;
  XtSetArg(arg[n], XtNhorizDistance, 5); n++;
  XtSetArg(arg[n], XtNfromHoriz, scroll_widget); n++;
  XtSetArg(arg[n], XtNjustify, XtJustifyRight); n++;
  XtSetArg(arg[n], XtNlabel, interval_string); n++;
  XtSetArg(arg[n], XtNborderWidth, (Dimension) 0); n++;
  interval_widget = XtCreateManagedWidget("interval", labelWidgetClass,
					   box_widget, arg, n);
  
  /* Create the scroll bar for the slowdown */

  n = 0;
  XtSetArg(arg[n], XtNvertDistance, 15); n++;
  XtSetArg(arg[n], XtNfromVert, title_widget); n++;
  XtSetArg(arg[n], XtNhorizDistance, 30); n++;
  XtSetArg(arg[n], XtNfromHoriz, interval_widget); n++;
  XtSetArg(arg[n], XtNorientation, XtorientHorizontal); n++;
  XtSetArg(arg[n], XtNlength, 100); n++;
  XtSetArg(arg[n], XtNthickness, 15); n++;
  scroll_widget2 = XtCreateManagedWidget("scroll2", scrollbarWidgetClass,
                                        box_widget, arg, n);
  XtAddCallback(scroll_widget2, XtNscrollProc, ScrollProc2, NULL);
  XtAddCallback(scroll_widget2, XtNjumpProc, JumpProc2, NULL);

  /* Create the label widget which displays the slowdown value
     associated with the scrollbar 2. */

  n = 0;
  (void) sprintf(slowdown_string, "%4d times", slowdown);
  XtSetArg(arg[n], XtNvertDistance, 10); n++;
  XtSetArg(arg[n], XtNfromVert, title_widget); n++;
  XtSetArg(arg[n], XtNhorizDistance, 5); n++;
  XtSetArg(arg[n], XtNfromHoriz, scroll_widget2); n++;
  XtSetArg(arg[n], XtNjustify, XtJustifyRight); n++;
  XtSetArg(arg[n], XtNlabel, slowdown_string); n++;
  XtSetArg(arg[n], XtNborderWidth, (Dimension) 0); n++;
  slowdown_widget = XtCreateManagedWidget("slowdown", labelWidgetClass,
                                           box_widget, arg, n);

  /* Now add the actual canvas ... pict_size square pixels*/

  n=0;
  XtSetArg(arg[n],XtNheight, pict_size); n++;
  XtSetArg(arg[n],XtNwidth,  pict_size); n++;
  XtSetArg(arg[n], XtNvertDistance, 10); n++;
  XtSetArg(arg[n], XtNfromVert, start_stop_button); n++;
  canvas_widget = XtCreateManagedWidget("canvas", compositeWidgetClass,
					box_widget, arg, n);

  /* Add callback for exposure */

  XtAddEventHandler(canvas_widget,ExposureMask,False,Exposed,NULL);

  /* Now add the color scale ... pict_size square pixels*/

  n=0;
  XtSetArg(arg[n],XtNheight, pict_size); n++;
  XtSetArg(arg[n],XtNwidth,  80); n++;
  XtSetArg(arg[n], XtNvertDistance, 10); n++;
  XtSetArg(arg[n], XtNfromVert, start_stop_button); n++;
  XtSetArg(arg[n], XtNhorizDistance, 20); n++;
  XtSetArg(arg[n], XtNfromHoriz, canvas_widget); n++;
  XtSetArg(arg[n], XtNborderWidth, 0); n++;
  map_widget = XtCreateManagedWidget("colorMap", compositeWidgetClass,
					box_widget, arg, n);

  /* Uncomment the next two lines to cause FileInput to be
     called whenever input is available on standard input */

  /*
    app_context = XtWidgetToApplicationContext(canvas_widget);
    (void) XtAppAddInput(app_context, 0,XtInputReadMask,FileInput,NULL);
  */

  /* Realize everything */

  XtRealizeWidget(top_level_widget);

  /* Set up the drawing environment */

  display = XtDisplay(canvas_widget);

  window = XtWindow(canvas_widget); 
  window_map = XtWindow(map_widget); 
  screen = DefaultScreen(display);
  visual = DefaultVisual(display, screen);
  depth = DisplayPlanes(display, screen);
  (void) printf("depth = %d\n",depth);
  gc = XCreateGC(display, window, 0, (XGCValues *) NULL); 

  gcv.font = XLoadFont(display, "8x13");
  if(!gcv.font)printf("error font not loaded\n");

  gc_map = XCreateGC(display, window_map, GCFont, &gcv); 
  
  Setcmap();

  /* Make image to match the size of our canvas */

  pict  = (u_char *) malloc((unsigned) (pict_size*pict_size));
  image = XCreateImage(display, visual, depth, ZPixmap, 0,
		       pict, pict_size, pict_size, 8, 0);


  /* Make the byte array which will hold the access data */

  if (!(grid = (u_char *) malloc((unsigned) (grid_size*grid_size))))
      Error("failed to allocate grid", -1);
  bzero((char *) grid, grid_size*grid_size);

  /* Make the byte array which will hold the access flag */

  if (!(flag = (u_char *) malloc((unsigned) (grid_size*grid_size))))
      Error("failed to allocate flag", -1);
  bzero((char *) flag, grid_size*grid_size);



  /* Make the array which will hold the integral */

  if (!(integr = (double *) malloc(sizeof(double)*(grid_size*grid_size))))
      Error("failed to allocate integr", -1);


  /* Make the array which will hold the last access time */

  if (!(ltime = (double *) malloc(sizeof(double)*(grid_size*grid_size))))
      Error("failed to allocate ltime", -1);

  for(i=0; i<grid_size*grid_size; i++, *ltime=0.,  *integr = 0.0);

  /* clear the array display */
  UpdatePixRegion(0, grid_size-1, 0, grid_size-1, 0,0.);

  /* Enter the event loop */

  XtMainLoop();

  return 0;  /* Never actually does this */
}

