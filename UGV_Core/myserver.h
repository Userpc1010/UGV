#ifndef MYSERVER_H
#define MYSERVER_H

#include <QTcpServer>
#include <QTcpSocket>
#include <QByteArray>

class TcpServer : public QObject
{
    Q_OBJECT

public:
    TcpServer(QObject *parent = nullptr);
    ~TcpServer();

signals:
    void Server_to_Controller (QByteArray point);

private slots:
    void new_Connection();
    void disconect();
    void read();

public slots:
    void Controller_to_Server(QByteArray data) { Write_to_Client(data); }

private:
    void Write_to_Client (QByteArray data);

    QTcpServer * server;
    QTcpSocket * socket;
    QByteArray arr;
    bool lock = false;
};

#endif // MYSERVER_H
