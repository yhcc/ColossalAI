""" Elasticsearch logging handler
"""

import datetime
import logging
import socket
from enum import Enum
from threading import Lock, Timer

from elasticsearch import Elasticsearch
from elasticsearch import helpers as eshelpers

from .serializers import CMRESSerializer


class CMRESHandler(logging.Handler):
    """Elasticsearch log handler

    Allows to log to elasticsearch into json format.
    All LogRecord fields are serialised and inserted
    """

    class AuthType(Enum):
        """Authentication types supported

        The handler supports
         - No authentication
         - Basic authentication
        """

        NO_AUTH = 0
        BASIC_AUTH = 1

    class IndexNameFrequency(Enum):
        """Index type supported
        the handler supports
        - Daily indices
        - Weekly indices
        - Monthly indices
        - Year indices
        """

        DAILY = 0
        WEEKLY = 1
        MONTHLY = 2
        YEARLY = 3

    # Defaults for the class
    __DEFAULT_ELASTICSEARCH_HOST = ["http://localhost:9200"]
    __DEFAULT_AUTH_USER = ""
    __DEFAULT_AUTH_PASSWD = ""
    __DEFAULT_AUTH_TYPE = AuthType.NO_AUTH
    __DEFAULT_INDEX_FREQUENCY = IndexNameFrequency.DAILY
    __DEFAULT_BUFFER_SIZE = 1000
    __DEFAULT_FLUSH_FREQ_INSEC = 0.1
    __DEFAULT_ADDITIONAL_FIELDS = {}
    __DEFAULT_ES_INDEX_NAME = "python_logger"
    __DEFAULT_ES_DOC_TYPE = "python_log"
    __DEFAULT_RAISE_ON_EXCEPTION = False
    __DEFAULT_TIMESTAMP_FIELD_NAME = "timestamp"

    __LOGGING_FILTER_FIELDS = ["msecs", "relativeCreated", "levelno", "created"]

    @staticmethod
    def _get_daily_index_name(es_index_name):
        """Returns elasticearch index name
        :param: index_name the prefix to be used in the index
        :return: A srting containing the elasticsearch indexname used which should include the date.
        """
        return "{0!s}-{1!s}".format(  # pylint: disable=C0209
            es_index_name, datetime.datetime.now().strftime("%Y.%m.%d")
        )

    @staticmethod
    def _get_weekly_index_name(es_index_name):
        """Return elasticsearch index name
        :param: index_name the prefix to be used in the index
        :return: A srting containing the elasticsearch indexname used which should include the date and specific week
        """
        current_date = datetime.datetime.now()
        start_of_the_week = current_date - datetime.timedelta(days=current_date.weekday())
        return "{0!s}-{1!s}".format(es_index_name, start_of_the_week.strftime("%Y.%m.%d"))  # pylint: disable=C0209

    @staticmethod
    def _get_monthly_index_name(es_index_name):
        """Return elasticsearch index name
        :param: index_name the prefix to be used in the index
        :return: A srting containing the elasticsearch indexname used which should include the date and specific moth
        """
        return "{0!s}-{1!s}".format(es_index_name, datetime.datetime.now().strftime("%Y.%m"))  # pylint: disable=C0209

    @staticmethod
    def _get_yearly_index_name(es_index_name):
        """Return elasticsearch index name
        :param: index_name the prefix to be used in the index
        :return: A srting containing the elasticsearch indexname used which should include the date and specific year
        """
        return "{0!s}-{1!s}".format(es_index_name, datetime.datetime.now().strftime("%Y"))  # pylint: disable=C0209

    _INDEX_FREQUENCY_FUNCION_DICT = {
        IndexNameFrequency.DAILY: _get_daily_index_name,
        IndexNameFrequency.WEEKLY: _get_weekly_index_name,
        IndexNameFrequency.MONTHLY: _get_monthly_index_name,
        IndexNameFrequency.YEARLY: _get_yearly_index_name,
    }

    def __init__(
        self,
        hosts=None,
        auth_details=(__DEFAULT_AUTH_USER, __DEFAULT_AUTH_PASSWD),
        auth_type=__DEFAULT_AUTH_TYPE,
        buffer_size=__DEFAULT_BUFFER_SIZE,
        flush_frequency_in_sec=__DEFAULT_FLUSH_FREQ_INSEC,
        es_index_name=__DEFAULT_ES_INDEX_NAME,
        index_name_frequency=__DEFAULT_INDEX_FREQUENCY,
        es_doc_type=__DEFAULT_ES_DOC_TYPE,
        es_additional_fields=None,
        raise_on_indexing_exceptions=__DEFAULT_RAISE_ON_EXCEPTION,
        default_timestamp_field_name=__DEFAULT_TIMESTAMP_FIELD_NAME,
    ):
        """Handler constructor

        :param hosts: The list of hosts that elasticsearch clients will connect. The list can be provided
                    in the format ```[{'host':'host1','port':9200}, {'host':'host2','port':9200}]``` to
                    make sure the client supports failover of one of the instertion nodes
        :param auth_details: When ```CMRESHandler.AuthType.BASIC_AUTH``` is used this argument must contain
                    a tuple of string with the user and password that will be used to authenticate against
                    the Elasticsearch servers, for example```('User','Password')
        :param auth_type: The authentication type to be used in the connection ```CMRESHandler.AuthType```
                    Currently, NO_AUTH, BASIC_AUTH, KERBEROS_AUTH are supported
        :param buffer_size: An int, Once this size is reached on the internal buffer results are flushed into ES
        :param flush_frequency_in_sec: A float representing how often and when the buffer will be flushed, even
                    if the buffer_size has not been reached yet
        :param es_index_name: A string with the prefix of the elasticsearch index that will be created. Note a
                    date with YYYY.MM.dd, ```python_logger``` used by default
        :param index_name_frequency: Defines what the date used in the postfix of the name would be. available values
                    are selected from the IndexNameFrequency class (IndexNameFrequency.DAILY,
                    IndexNameFrequency.WEEKLY, IndexNameFrequency.MONTHLY, IndexNameFrequency.YEARLY). By default
                    it uses daily indices.
        :param es_doc_type: A string with the name of the document type that will be used ```python_log``` used
                    by default
        :param es_additional_fields: A dictionary with all the additional fields that you would like to add
                    to the logs, such the application, environment, etc.
        :param raise_on_indexing_exceptions: A boolean, True only for debugging purposes to raise exceptions
                    caused when
        :return: A ready to be used CMRESHandler.
        """
        logging.Handler.__init__(self)
        if hosts is None:
            hosts = self.__DEFAULT_ELASTICSEARCH_HOST

        if es_additional_fields is None:
            es_additional_fields = self.__DEFAULT_ADDITIONAL_FIELDS

        self.hosts = hosts
        self.auth_details = auth_details
        self.auth_type = auth_type
        self.buffer_size = buffer_size
        self.flush_frequency_in_sec = flush_frequency_in_sec
        self.es_index_name = es_index_name
        self.index_name_frequency = index_name_frequency
        self.es_doc_type = es_doc_type
        self.es_additional_fields = es_additional_fields.copy()
        self.es_additional_fields.update(
            {"host": socket.gethostname(), "host_ip": socket.gethostbyname(socket.gethostname())}
        )
        self.raise_on_indexing_exceptions = raise_on_indexing_exceptions
        self.default_timestamp_field_name = default_timestamp_field_name

        self._client = None
        self._buffer = []
        self._buffer_lock = Lock()
        self._timer = None
        self._index_name_func = CMRESHandler._INDEX_FREQUENCY_FUNCION_DICT[self.index_name_frequency]
        self.serializer = CMRESSerializer()

    def __schedule_flush(self):
        if self._timer is None:
            self._timer = Timer(self.flush_frequency_in_sec, self.flush)
            self._timer.setDaemon(True)
            self._timer.start()

    def __get_es_client(self):
        if self.auth_type == CMRESHandler.AuthType.NO_AUTH:
            if self._client is None:
                self._client = Elasticsearch(
                    hosts=self.hosts,
                    serializer=self.serializer,
                    request_timeout=30,
                )
            return self._client

        if self.auth_type == CMRESHandler.AuthType.BASIC_AUTH:
            if self._client is None:
                return Elasticsearch(
                    hosts=self.hosts,
                    http_auth=self.auth_details,
                    serializer=self.serializer,
                    request_timeout=30,
                )
            return self._client

        raise ValueError("Authentication method not supported")

    def test_es_source(self):
        """Returns True if the handler can ping the Elasticsearch servers

        Can be used to confirm the setup of a handler has been properly done and confirm
        that things like the authentication is working properly

        :return: A boolean, True if the connection against elasticserach host was successful
        """
        return self.__get_es_client().ping()

    @staticmethod
    def __get_es_datetime_str(timestamp):
        """Returns elasticsearch utc formatted time for an epoch timestamp

        :param timestamp: epoch, including milliseconds
        :return: A string valid for elasticsearch time record
        """
        current_date = datetime.datetime.utcfromtimestamp(timestamp)
        return "{0!s}.{1:03d}Z".format(  # pylint: disable=C0209
            current_date.strftime("%Y-%m-%dT%H:%M:%S"), int(current_date.microsecond / 1000)
        )

    def flush(self):
        """Flushes the buffer into ES
        :return: None
        """
        if self._timer is not None and self._timer.is_alive():
            self._timer.cancel()
        self._timer = None

        if self._buffer:
            try:
                with self._buffer_lock:
                    logs_buffer = self._buffer
                    self._buffer = []

                eshelpers.bulk(
                    client=self.__get_es_client(),
                    actions=logs_buffer,
                    stats_only=True,
                    index=self._index_name_func.__func__(self.es_index_name),
                )
            except Exception as exception:  # pylint: disable=W0703
                if self.raise_on_indexing_exceptions:
                    raise exception

    def close(self):
        """Flushes the buffer and release any outstanding resource

        :return: None
        """
        if self._timer is not None:
            self.flush()
        self._timer = None

    def emit(self, record):
        """Emit overrides the abstract logging.Handler logRecord emit method

        Format and records the log

        :param record: A class of type ```logging.LogRecord```
        :return: None
        """
        self.format(record)

        rec = self.es_additional_fields.copy()
        for key, value in record.__dict__.items():
            if key not in CMRESHandler.__LOGGING_FILTER_FIELDS:
                if key == "args":
                    value = tuple(str(arg) for arg in value)
                rec[key] = "" if value is None else value
        rec[self.default_timestamp_field_name] = self.__get_es_datetime_str(record.created)
        with self._buffer_lock:
            self._buffer.append(rec)

        if len(self._buffer) >= self.buffer_size:
            self.flush()
        else:
            self.__schedule_flush()
