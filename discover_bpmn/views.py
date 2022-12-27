from rest_framework.decorators import api_view
from rest_framework.response import Response
import pygraphviz as pgv
import itertools
from collections import defaultdict
from typing import Dict, Set
import pandas as pd
from itertools import chain
from more_itertools import pairwise
from collections import Counter
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.decorators import parser_classes
import os
import pm4py
from pm4py.visualization.bpmn import visualizer as pn_visualizer
from pm4py.objects.conversion.log import converter as xes_converter
from pm4py.objects.log.importer.xes import importer as xes_importer

@api_view(['POST'])
@parser_classes([FormParser, MultiPartParser])
def discover_model(request):
    file_extension = get_file_extension(request.data['file'].name)

    if request.data['algorithm'] == 'own':
        if file_extension == '.csv':
            traces = csv_file_to_traces(
                request.data['file'],
                id_column_name=request.data['id'],
                activity_column_name=request.data['activity'],
                timestamp_column_name=request.data['timestamp']
            )
        elif file_extension == '.xes':
            traces = xes_file_to_traces(request.data['file'])
        print(traces)
        graph = alpha_draw_model(traces)

    elif request.data['algorithm'] == 'pm4py':
        if file_extension == '.csv':
            dataframe = pd.read_csv(request.data['file'])
            dataframe = pm4py.format_dataframe(
                dataframe,
                case_id='Case ID',
                activity_key='Activity',
                timestamp_key='Start Timestamp'
            )
            event_log = pm4py.convert_to_event_log(dataframe)
        elif file_extension == '.xes':
            file = request.data['file']
            with open('pm.xes', 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)

            event_log = pm4py.read_xes("pm.xes")
            os.remove("pm.xes")
            print(event_log)

        net = pm4py.discover_bpmn_inductive(event_log)
        graph = pn_visualizer.apply(net).source

    return Response({"graph": graph})

def get_file_extension(filename):
    return os.path.splitext(filename)[1]

def csv_file_to_traces(file, id_column_name, activity_column_name, timestamp_column_name):
    log = (pd.read_csv(file)
           .sort_values(by=[id_column_name, timestamp_column_name])
           .groupby([id_column_name])
           .agg({activity_column_name: lambda x: tuple(x)}))

    traces = []
    for log_entry in log[activity_column_name]:
        traces.append(log_entry)

    return traces

def xes_file_to_traces(file):
    with open('pm.xes', 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)

    event_log = pm4py.read_xes("pm.xes")
    os.remove("pm.xes")

    log = (event_log
           .sort_values(by=['case:concept:name', 'time:timestamp'])
           .groupby(['case:concept:name'])
           .agg({'Activity': lambda x: tuple(x)}))

    traces = []
    for log_entry in log['Activity']:
        traces.append(log_entry)

    return traces


def alpha_draw_model(traces):
    dfg, event_counter = get_trace_and_flow_counts(traces)
    G = MyGraph()

    start_set_events = get_start_events(traces)
    end_set_events = get_end_events(traces)
    direct_succession = get_direct_succession(traces)
    parallel_events = get_potential_parallelism(direct_succession)
    causality = get_causality(direct_succession)
    inv_causality = get_inv_causality(causality)

    # adding simple relation
    # IMPORTANT: in get_inv_casuality there are no sets with only one value
    for event in causality:
        if (len(causality[event]) == 1
                and list(causality[event])[0] not in inv_causality
                and event not in end_set_events):
            G.add_edge((event, event_counter[event]),
                       (list(causality[event])[0], event_counter[list(causality[event])[0]]))

    # adding split gateways based on causality
    for event in causality:
        if len(causality[event]) > 1 and not list(causality[event])[0] in inv_causality:
            if should_draw_and_gateway(event, causality, parallel_events):
                G.add_and_split_gateway(dfg, event_counter, event, causality[event])
            else:
                G.add_xor_split_gateway(dfg, event_counter, event, causality[event])

    # adding merge gateways based on inverted causality
    for event in inv_causality:
        if should_draw_merge_split_gateway(event, causality, inv_causality):
            if should_draw_and_gateway(event, inv_causality, parallel_events):
                G.add_and_merge_gateway(dfg, event_counter, inv_causality[event], event)
            else:
                G.add_xor_merge_gateway(dfg, event_counter, inv_causality[event], event)
        elif not should_draw_merge_split_gateway(event, causality, inv_causality):
            G.add_xor_merge_and_split_gateway(
                dfg, event_counter, inv_causality[event], causality[list(inv_causality[event])[0]]
            )
        elif len(inv_causality[event]) == 1:
            source = list(inv_causality[event])[0]
            G.add_edge((source, event_counter[source]), (event, event_counter[event]))

    # adding start event
    G.add_event("start")
    if len(start_set_events) > 1:
        if tuple(start_set_events) in parallel_events:
            G.add_and_split_gateway(dfg, event_counter, "start", start_set_events)
        else:
            G.add_xor_split_gateway(dfg, event_counter, "start", start_set_events)
    else:
        G.add_edge("start", (list(start_set_events)[0], event_counter[list(start_set_events)[0]]))

    # adding end event
    G.add_end_event("end")
    if len(end_set_events) > 1:
        if tuple(end_set_events) in parallel_events:
            G.add_and_merge_gateway(dfg, event_counter, end_set_events, "end")
        else:
            G.add_xor_merge_gateway(dfg, event_counter, end_set_events, "end")
    else:
        if (list(end_set_events)[0] not in causality):
            G.add_edge((list(end_set_events)[0], event_counter[list(end_set_events)[0]]), "end")
        else:
            end_node_successors = set(causality[list(end_set_events)[0]])
            end_node_successors.add("end")
            G.add_xor_split_gateway(dfg, event_counter, list(end_set_events)[0], end_node_successors)

    return G.string()


def get_start_events(traces):
    start_events = set()
    for trace in traces:
        start_events.add(trace[0])

    return start_events


def get_end_events(traces):
    end_events = set()
    for trace in traces:
        end_events.add(trace[-1])

    return end_events


def get_direct_succession(traces):
    direct_succession = defaultdict(set)
    for trace in traces:
        for (event_index, event) in enumerate(trace):
            if (event_index != len(trace) - 1):
                direct_succession[event].add(trace[event_index + 1])

    return direct_succession


def get_potential_parallelism(direct_succession):
    pottential_parallelism = set([])
    for event, events_successors in direct_succession.items():
        for events_successor in events_successors:
            if events_successor in direct_succession and event in direct_succession[events_successor]:
                pottential_parallelism.add((event, events_successor))

    return pottential_parallelism


def get_causality(direct_succession) -> Dict[str, Set[str]]:
    causality = defaultdict(set)
    for ev_cause, events in direct_succession.items():
        for event in events:
            if ev_cause not in direct_succession.get(event, set()):
                causality[ev_cause].add(event)

    return dict(causality)


def get_inv_causality(causality) -> Dict[str, Set[str]]:
    inv_causality = defaultdict(set)
    for key, values in causality.items():
        for value in values:
            inv_causality[value].add(key)

    return {k: v for k, v in inv_causality.items() if len(v) > 1}


def should_draw_merge_split_gateway(event, casuality, inv_causality):
    should_draw_merge_gateway = len(inv_causality[event]) > 1
    for predecessor_event in inv_causality[event]:
        if (len(casuality[predecessor_event]) != 1):
            should_draw_merge_gateway = False

    return should_draw_merge_gateway


def find_subsets(s, n):
    return list(itertools.combinations(s, n))


def should_draw_and_gateway(event, casuality_or_inv_causality, parallel_events):
    casuality_subsets = find_subsets(casuality_or_inv_causality[event], 2)
    should_draw_and_split_gateway = True
    for subset in casuality_subsets:
        if subset not in parallel_events:
            should_draw_and_split_gateway = False
            break

    return should_draw_and_split_gateway


def get_trace_and_flow_counts(A_traces):
    df = list(chain.from_iterable(A_traces))
    df = pd.DataFrame([df]).T
    df.columns = ['Event']
    df_traces = pd.DataFrame([A_traces]).T
    df_traces.columns = ['Traces']
    dfs_traces_count = df_traces.groupby("Traces")["Traces"].count()
    ev_counter = df.Event.value_counts()
    color_min = ev_counter.min()
    color_max = ev_counter.max()
    dfg = dict()
    ev_start_set = set()
    ev_end_set = set()
    for trace, count in dfs_traces_count.items():
        if trace[0] not in ev_start_set:
            ev_start_set.add(trace[0])
        if trace[-1] not in ev_end_set:
            ev_end_set.add(trace[-1])
        for ev_i, ev_j in pairwise(trace):
            if ev_i not in dfg.keys():
                dfg[ev_i] = Counter()
            dfg[ev_i][ev_j] += count

    return dfg, ev_counter


class MyGraph(pgv.AGraph):
    def __init__(self, *args):
        super(MyGraph, self).__init__(strict=False, directed=True, *args)
        self.graph_attr['rankdir'] = 'LR'
        self.node_attr['shape'] = 'Mrecord'
        self.graph_attr['splines'] = 'ortho'
        self.graph_attr['nodesep'] = '0.8'
        self.edge_attr.update(penwidth='2')

    def add_event(self, name):
        super(MyGraph, self).add_node(name, shape="circle", label="")

    def add_end_event(self, name):
        super(MyGraph, self).add_node(name, shape="circle", label="", penwidth='3')

    def add_and_gateway(self, *args):
        super(MyGraph, self).add_node(*args, shape="diamond",
                                      width=".7", height=".7",
                                      fixedsize="true",
                                      fontsize="40", label="+")

    def add_xor_gateway(self, *args, **kwargs):
        super(MyGraph, self).add_node(*args, shape="diamond",
                                      width=".7", height=".7",
                                      fixedsize="true",
                                      fontsize="40", label="×")

    def add_and_split_gateway(self, dfg, event_counter, source, targets, *args):
        gateway = 'ANDs ' + str(source) + '->' + str(targets)
        self.add_and_gateway(gateway, *args)
        if (source != 'start'):
            super(MyGraph, self).add_edge((source, event_counter[source]), gateway)
        else:
            super(MyGraph, self).add_edge(source, gateway)
        for target in targets:
            if (target != 'end' and source != 'start'):
                super(MyGraph, self).add_edge(gateway, (target, event_counter[target]), label=dfg[source][target])
            elif (target != 'end'):
                super(MyGraph, self).add_edge(gateway, (target, event_counter[target]))
            else:
                super(MyGraph, self).add_edge(gateway, target)

    def add_xor_split_gateway(self, dfg, event_counter, source, targets, *args):
        gateway = 'XORs ' + str(source) + '->' + str(targets)
        self.add_xor_gateway(gateway, *args)
        if (source != 'start'):
            super(MyGraph, self).add_edge((source, event_counter[source]), gateway)
        else:
            super(MyGraph, self).add_edge(source, gateway)
        for target in targets:
            if (target != 'end' and source != 'start'):
                super(MyGraph, self).add_edge(gateway, (target, event_counter[target]), label=dfg[source][target])
            elif (target != 'end'):
                super(MyGraph, self).add_edge(gateway, (target, event_counter[target]))
            else:
                super(MyGraph, self).add_edge(gateway, target)

    def add_and_merge_gateway(self, dfg, event_counter, sources, target, *args):
        gateway = 'ANDm ' + str(sources) + '->' + str(target)
        self.add_and_gateway(gateway, *args)
        if (target != 'end'):
            super(MyGraph, self).add_edge(gateway, (target, event_counter[target]))
        else:
            super(MyGraph, self).add_edge(gateway, target)
        for source in sources:
            if (source != 'start' and target != 'end'):
                super(MyGraph, self).add_edge((source, event_counter[source]), gateway, label=dfg[source][target])
            elif (source != 'start'):
                super(MyGraph, self).add_edge((source, event_counter[source]), gateway)
            else:
                super(MyGraph, self).add_edge(source, gateway)

    def add_xor_merge_gateway(self, dfg, event_counter, sources, target, *args):
        gateway = 'XORm ' + str(sources) + '->' + str(target)
        self.add_xor_gateway(gateway, *args)
        if (target != 'end'):
            super(MyGraph, self).add_edge(gateway, (target, event_counter[target]))
        else:
            super(MyGraph, self).add_edge(gateway, target)
        for source in sources:
            if (source != 'start' and target != 'end'):
                super(MyGraph, self).add_edge((source, event_counter[source]), gateway, label=dfg[source][target])
            elif (source != 'start'):
                super(MyGraph, self).add_edge((source, event_counter[source]), gateway)
            else:
                super(MyGraph, self).add_edge(source, gateway)

    def add_xor_merge_and_split_gateway(self, dfg, event_counter, sources, targets, *args):
        xor_merge_gateway = 'XORm ' + str(sources) + '->' + 'ANDs '
        and_split_gateway = 'ANDs ' + 'XORm ' + '->' + str(targets)

        if (f"{xor_merge_gateway}" not in super(MyGraph, self).nodes()):

            self.add_xor_gateway(xor_merge_gateway, *args)
            self.add_and_gateway(and_split_gateway, *args)

            super(MyGraph, self).add_edge(xor_merge_gateway, and_split_gateway)

            for source in sources:
                if (source != 'start'):
                    super(MyGraph, self).add_edge((source, event_counter[source]), xor_merge_gateway)
                else:
                    super(MyGraph, self).add_edge(source, xor_merge_gateway)

            for target in targets:
                if (target != 'end'):
                    super(MyGraph, self).add_edge(and_split_gateway, (target, event_counter[target]))
                else:
                    super(MyGraph, self).add_edge(and_split_gateway, target)