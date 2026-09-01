#include "myserver.h"
#include <QMessageBox>
#include <QDebug>

TcpServer::TcpServer(QObject *parent)
    : QObject(parent)
{
    server = new QTcpServer(this);

    if (server->listen(QHostAddress::Any, 8981)) {
        qDebug() << "Сервер запущен";
    } else {
        QMessageBox::critical(0, "Ошибка сервера",
                              "Старт невозможен: " + server->errorString());
        server->close();
    }

    connect(server, SIGNAL(newConnection()), this, SLOT(new_Connection()));
}

TcpServer::~TcpServer()
{
    lock = false;
    if (socket) {
        socket->close();
        socket->deleteLater();
    }
    server->close();
    server->deleteLater();
}

void TcpServer::new_Connection()
{
    socket = server->nextPendingConnection();

    connect(socket, SIGNAL(disconnected()), this, SLOT(disconect()));
    connect(socket, SIGNAL(readyRead()), this, SLOT(read()));

    socket->write("Connection");
    socket->waitForBytesWritten(5000);

    qDebug() << "Соединение установлено";

    lock = true;
}

void TcpServer::disconect()
{
    lock = false;
    if (socket->isOpen()) {
        socket->close();
    }
    socket->deleteLater();
}

void TcpServer::read()
{
    arr = socket->readAll();

    // Пересылаем только то, что реально приходит от клиента
    if (arr.size() >= 3) {
        if ((arr[0] == 'M' && arr[1] == 'A' && arr[2] == 'N') ||   // ручное управление
            (arr[0] == 'M' && arr[1] == 'O' && arr[2] == 'D' && arr[3] == 'E')) {  // смена режима
            emit Server_to_Controller(arr);
        }
    }
}

void TcpServer::Write_to_Client(QByteArray data)
{
    if (lock) {
        socket->write(data);
        socket->waitForBytesWritten(5000);
        socket->flush();
    }
}
