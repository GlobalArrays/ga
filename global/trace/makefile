CFLAGS = -g

default: adjust collisions

adjust: adjust.c 
	$(CC) $(CFLAGS) -o $@ adjust.c

collisions: collisions.c 
	$(CC) $(CFLAGS) -o $@ collisions.c

clean:
	rm -f adjust collisions
