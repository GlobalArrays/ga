
int  poll_server(void);
void send_to_server(void *, int);
void recv_from_server(void *, int *);
int server_init(int, char **);
void server_close(void);
int server_clusid(void);

