from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import socket
import threading
from slack import WebClient
from slack.errors import SlackApiError

SLACK_MAX_PRINT_ERROR = 3
SLACK_ERROR_CODE = {'not_active': 1,
                    'API': 2}


def welcome_message():
    hostname = socket.gethostname()
    all_args = ' '.join(sys.argv)
    out_text = 'On server {}: {}\n'.format(hostname, all_args)
    return out_text


class Notifier(object):


    def __init__(self, token_file):

        self.active = False
        self.thread_id = None
        self.counter = 0  # count number of errors during Web API call
        self.client = None
        self.token = None
        self.channel = None

        if not os.path.exists(token_file):
            print('[SLACK] token file not found. You will not be notified.')
            return

        try:
            with open(token_file, 'r') as f:
                lines = [line.strip() for line in f.readlines()]

            if len(lines) >= 2:
                self.token = lines[0]
                self.channel = lines[1]
                self.client = WebClient(token=self.token)
                self.active = True
            else:
                print('[SLACK] token file format incorrect. Expected at least 2 lines.')

        except Exception as e:
            print(e)
            print('[SLACK] fail to read token file. You will not be notified.')

    def _handel_error(self, e):
        assert e.response["ok"] is False
        assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
        self.counter += 1
        if self.counter <= SLACK_MAX_PRINT_ERROR:
            print(f"Got the following error, you will not be notified: {e.response['error']}")

    def send_init_text(self, text=None):

        if not self.active:
            return SLACK_ERROR_CODE['not_active']

        if text is None:
            text = welcome_message()

        try:
            response = self.client.chat_postMessage(channel=self.channel, text=text)
            self.thread_id = response['ts']
            print('[SLACK] sent initial text. Chat ID %s. Message %s' % (self.thread_id, text))
            return 0
        except SlackApiError as e:
            self._handel_error(e)
            return SLACK_ERROR_CODE['API']

    def send_init_file(self, file_path, title=''):

        if not self.active:
            return SLACK_ERROR_CODE['not_active']

        try:
            # Fixed: Used self.client instead of undefined 'sc'
            response = self.client.files_upload(title=title, channels=self.channel, file=file_path)
            self.thread_id = response['ts']
            print('[SLACK] sent initial file. Chat ID %s.' % self.thread_id)
            return 0
        except SlackApiError as e:
            self._handel_error(e)
            return SLACK_ERROR_CODE['API']

    def send_text(self, text, reply_broadcast=False):

        print(text)
        if not self.active:
            return SLACK_ERROR_CODE['not_active']

        if self.thread_id is None:
            self.send_init_text(text)
        else:
            try:
                self.client.chat_postMessage(
                    channel=self.channel,
                    text=text,
                    thread_ts=self.thread_id,
                    as_user=True,
                    reply_broadcast=reply_broadcast
                )
            except SlackApiError as e:
                self._handel_error(e)
                return SLACK_ERROR_CODE['API']
        return 0

    def _send_file(self, file_path, title='', reply_broadcast=False):
        try:
            self.client.files_upload(
                title=title,
                channels=self.channel,
                thread_ts=self.thread_id,
                file=file_path,
                reply_broadcast=reply_broadcast
            )
            return 0
        except SlackApiError as e:
            self._handel_error(e)
            return SLACK_ERROR_CODE['API']

    def send_file(self, file_path, title='', reply_broadcast=False):
        if not self.active:
            return SLACK_ERROR_CODE['not_active']

        if self.thread_id is None:
            return self.send_init_file(file_path, title)
        else:
            os_thread = threading.Thread(
                target=self._send_file,
                args=(file_path, title, reply_broadcast)
            )
            os_thread.start()
        return 0
