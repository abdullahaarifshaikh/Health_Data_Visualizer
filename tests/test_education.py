import io
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
import app as web
from core.analytics import relationship, distribution, direct_relationship
from core.education import build_graph, prompt_context, unpack, planning_messages

def test_plan(state):
    question = state["question"].lower()
    chart = any(word in question for word in ("relates", "relationship", "compare all three", "chart", "matrix"))
    calculation = chart or any(word in question for word in ("mean", "average", "summarize", "table"))
    return {"intent":"relationship" if chart else "lookup" if calculation else "concept", "goal":state["question"], "steps":[], "requires_calculation":calculation, "requires_chart":chart}

from core.storage import Workspace
from core.tools import ToolContext, scatter_plot, correlation_heatmap, histogram

class Model:
    def __init__(self, responses): self.responses, self.calls = iter(responses), []
    def bind_tools(self, tools, **kwargs): return self
    def invoke(self, messages):
        self.calls.append(messages)
        response = next(self.responses)
        if isinstance(response, Exception): raise response
        return response

def call(name, args, id='test-call'):
    return AIMessage(content='', tool_calls=[{'name':name, 'args':args, 'id':id, 'type':'tool_call'}])

def answer(text, suggestions=None):
    return AIMessage(content=json.dumps({'answer':text, 'followups':suggestions or []}))

class TutorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='artifacts')
        self.workspace = Workspace('unit_test_workspace_123456', root=self.tmp.name, remote='')
        self.frame = pd.DataFrame({'heart_rate_bpm':[90,80,70,60,75], 'sleep_hours':[5,6,7,8,None], 'age_years':[20,30,40,50,60], 'group':['A','A','B','B','A']})
        self.context = ToolContext(self.frame, self.workspace, 'ds', 'chat')
        self.config = {'configurable':{'thread_id':'ds:chat'}}
    def tearDown(self):
        self.workspace.close(); self.tmp.cleanup()
    def run_graph(self, model, question='What relates heart rate and sleep hours?'):
        return build_graph(model, self.workspace.checkpointer, planner=test_plan, finalizer=lambda state,payload: dict(zip(("answer","followups"),unpack(model.invoke([SystemMessage(content='Earlier context summaries: ' + json.dumps(payload['earlier_context']))] + prompt_context(state)[0]))))).invoke({'question':question}, self.config, context=self.context)
    def relationship_model(self, final=None):
        return Model([call('calculate_statistics', {'operation':'pearson','columns':['heart_rate_bpm','sleep_hours']},'correlation'),
          call('scatter_plot', {'x':'heart_rate_bpm','y':'sleep_hours'},'scatter'),
          final or answer('The calculated Pearson correlation is -1 using four complete pairs; one record was excluded. This association does not establish causation.', ['How does the heart rate pattern compare between groups?'])])
    def test_ai_selects_calculation_and_chart_separately(self):
        model = self.relationship_model()
        state = self.run_graph(model)
        self.assertFalse(state['result']['partial'])
        self.assertEqual(state['result']['charts'][0]['type'],'scatter')
        self.assertEqual(state['result']['evidence'][0]['pearson_r'],-1)
        self.assertEqual(len(model.calls),3)
        self.assertNotIn('frame', state)
        self.assertTrue(state['result']['followups'])
    def test_calculation_does_not_automatically_draw(self):
        state = self.run_graph(Model([call('calculate_statistics', {'operation':'mean','columns':['age_years']}),answer('The calculated mean age is 40.')]),'Average age?')
        self.assertEqual(state['result']['charts'],[])
        self.assertEqual(state['result']['tables'][0]['rows'][0][2],40)
    def test_tool_error_can_be_corrected(self):
        model = Model([call('calculate_statistics',{'operation':'mean','columns':['unknown']}),
                       call('calculate_statistics',{'operation':'mean','columns':['age_years']},'corrected'),answer('The calculated mean age is 40.')])
        state = self.run_graph(model,'Average age?')
        self.assertFalse(state['result']['partial'])
        self.assertTrue(any(isinstance(m,ToolMessage) and m.status=='error' for m in state['messages']))
    def test_matrix_heatmap_and_result_reference(self):
        class MatrixModel:
            def bind_tools(self,t,**kwargs): return self
            def invoke(self,m):
                last = next(v for v in reversed(m) if isinstance(v,(HumanMessage,ToolMessage)))
                if isinstance(last,HumanMessage):
                    return call('calculate_statistics',{'operation':'spearman','columns':['heart_rate_bpm','sleep_hours','age_years']})
                facts=json.loads(last.content)
                if 'matrix' in facts: return call('correlation_heatmap',{'source_result_id':facts['result_id']},'heat')
                return answer('These are pairwise rank correlations; they do not account for the other variables.')
        state=self.run_graph(MatrixModel(),'Compare all three numeric measurements.')
        self.assertEqual(state['result']['charts'][0]['type'],'heatmap')
        self.assertEqual(len(state['result']['evidence'][0]['pairs']),3)
    def test_query_table_to_bar_chart(self):
        class QueryModel:
            def bind_tools(self,t,**kwargs): return self
            def invoke(self,m):
                last = next(v for v in reversed(m) if isinstance(v,(HumanMessage,ToolMessage)))
                if isinstance(last,HumanMessage): return call('query_data',{'sql':'SELECT "group", avg(age_years) AS mean_age FROM healthcare GROUP BY "group"'})
                facts=json.loads(last.content)
                if facts['operation']=='query': return call('bar_chart',{'source_result_id':facts['result_id'],'x':'group','y':'mean_age'},'bar')
                return answer('The bar chart compares the computed mean ages by group.')
        state=self.run_graph(QueryModel(),'Show mean age by group in a bar chart.')
        self.assertEqual(state['result']['charts'][0]['type'],'bar')
    def test_filters_missing_and_constant(self):
        result=relationship(pd.DataFrame({'x':[1,1,np.inf],'y':[2,3,4]}),'x','y')
        self.assertIsNone(result['facts']['pearson_r'])
        state=self.run_graph(Model([call('calculate_statistics',{'operation':'mean','columns':['age_years'],'filters':[{'column':'age_years','op':'gt','value':30}]}),answer('The filtered mean age is 50.')]),'Mean age above 30?')
        self.assertEqual(state['result']['tables'][0]['rows'][0][2],50)
    def test_grounding_replaces_invented_per_unit_effect(self):
        result=self.run_graph(self.relationship_model(answer('Heart rate is about 0.1 bpm less for each extra hour of sleep.')))['result']
        self.assertEqual(result['partial_reason'],'grounding')
        self.assertNotIn('0.1 bpm less',result['answer'])
    def test_loop_is_bounded(self):
        state=self.run_graph(Model([call('inspect_dataset',{},str(i)) for i in range(6)]),'Tell me about the table.')
        self.assertTrue(state['result']['partial'])
        self.assertLessEqual(state['tool_rounds'],5)
        self.assertIsInstance(state['messages'][-1],AIMessage)
    def test_followups_are_contextual_and_deduplicated(self):
        q='How does the heart rate pattern compare between groups?'
        first=self.run_graph(self.relationship_model(answer('The correlation describes heart rate and sleep hours.',[q,q,'Can you perform regression for diabetes?'])))
        self.assertEqual(first['result']['followups'],[q])
        second=self.run_graph(Model([answer('This describes the recorded heart rate and sleep hours.',[q,'How many sleep hours values were missing?'])]),'Explain the last answer.')
        self.assertEqual(second['result']['followups'],['How many sleep hours values were missing?'])
    def test_native_checkpoint_survives_reopen_and_retains_more_than_four_turns(self):
        self.run_graph(self.relationship_model())
        for i in range(6): self.run_graph(Model([answer(f'Concept explanation number {i}.')]),f'Concept question {i}?')
        self.workspace.close()
        self.workspace=Workspace('unit_test_workspace_123456',root=self.tmp.name,remote='')
        self.context.workspace=self.workspace
        model=Model([answer('The earlier calculated correlation used four complete pairs.')])
        state=self.run_graph(model,'What was the heart rate correlation we calculated at the beginning?')
        self.assertEqual(sum(isinstance(m,HumanMessage) for m in state['messages']),8)
        prompt=' '.join(str(m.content) for m in model.calls[0])
        self.assertIn('pearson_r',prompt)
        self.assertIn('Earlier context summaries',prompt)
        self.assertNotIn('frame',state)
        self.assertNotIn('result', build_graph(model, self.workspace.checkpointer).get_state(self.config).values)
    def test_results_are_isolated_between_chats(self):
        state=self.run_graph(self.relationship_model())
        key=next(m.artifact['result_id'] for m in state['messages'] if isinstance(m,ToolMessage) and m.status!='error')
        with self.assertRaises(ValueError): self.workspace.artifact(key,'ds','another-chat')
    def test_external_sql_access_is_blocked(self):
        state=self.run_graph(Model([call('query_data',{'sql':"SELECT * FROM read_csv_auto('C:/Windows/win.ini')"}),answer('The requested external file cannot be accessed.')]))
        self.assertTrue(any(isinstance(m,ToolMessage) and m.status=='error' for m in state['messages']))

    def test_numeric_summary_operations_and_grouping(self):
        expected = {'count': 5, 'sum': 200, 'mean': 40, 'median': 40, 'min': 20,
                    'max': 60, 'std': np.sqrt(250), 'variance': 250}
        for operation, value in expected.items():
            self.config = {'configurable': {'thread_id': 'ds:' + operation}}
            result = self.run_graph(Model([call('calculate_statistics', {'operation': operation, 'columns': ['age_years']}),
                                           answer('The requested calculation completed.')]), 'Summarize age.')['result']
            self.assertAlmostEqual(result['tables'][0]['rows'][0][2], value, places=6)
        self.config = {'configurable': {'thread_id': 'ds:groups'}}
        result = self.run_graph(Model([call('calculate_statistics', {'operation': 'mean', 'columns': ['age_years'], 'group_by': 'group'}),
                                       answer('The calculated means differ by group.')]), 'Mean age by group?')['result']
        self.assertAlmostEqual(result['tables'][0]['rows'][0][2], 110 / 3)

    def test_timeout_preserves_computed_evidence_and_discloses_partial(self):
        with self.assertLogs('core.errors', level='WARNING'):
            result = self.run_graph(Model([call('calculate_statistics', {'operation': 'pearson', 'columns': ['heart_rate_bpm', 'sleep_hours']}),
                                           TimeoutError('simulated provider timeout')]))['result']
        self.assertTrue(result['partial'])
        self.assertIn('r = -1.000', result['answer'])
        self.assertEqual(result['charts'], [])

    def test_answer_does_not_duplicate_suggestions_or_invent_image_links(self):
        response = answer('Explanation about age.\n\n![chart](attachment://fake.png)\n\n**Follow-up suggestions**\n1. What is the age range?',
                          ['What is the age range?'])
        result = self.run_graph(Model([response]), 'Explain age.')['result']
        self.assertNotIn('attachment://', result['answer'])
        self.assertNotIn('Follow-up suggestions', result['answer'])
        self.assertEqual(result['followups'], ['What is the age range?'])

    def test_completion_gate_prevents_finishing_relationship_without_chart(self):
        model = Model([call('calculate_statistics', {'operation': 'pearson', 'columns': ['heart_rate_bpm', 'sleep_hours']}),
                       answer('The correlation is -1.'),
                       call('scatter_plot', {'x': 'heart_rate_bpm', 'y': 'sleep_hours'}, 'finish-chart'),
                       answer('The correlation is -1 using four complete pairs. One record was excluded. This is descriptive association.')])
        state = self.run_graph(model)
        self.assertEqual(state['result']['charts'][0]['type'], 'scatter')
        self.assertFalse(state['completion_missing'])
        self.assertIn('check_completion', state['trace'])
        self.assertTrue(any('planned work is incomplete' in str(m.content) for m in model.calls[2]))

    def test_greeting_bypasses_planning_models_and_tools(self):
        model = Model([])
        with patch('core.education.make_model', side_effect=AssertionError('A greeting must not call a model')):
            state = self.run_graph(model, 'Hi!')
        self.assertEqual(model.calls, [])
        self.assertEqual(state['model_calls'], 0)
        self.assertEqual(state['result']['charts'], [])
        self.assertNotIn('plan_analysis', state['trace'])
        self.assertNotIn('check_completion', state['trace'])
        self.assertFalse(state['result']['partial'])
        self.assertEqual(state['result']['teaching_note'], '')

    def test_greeting_with_a_data_request_does_not_bypass_analysis(self):
        state = self.run_graph(Model([call('calculate_statistics', {'operation': 'mean', 'columns': ['age_years']}),
                                      answer('The mean age is 40.')]), 'Hi, what is the average age?')
        self.assertIn('plan_analysis', state['trace'])
        self.assertTrue(state['result']['evidence'])

    def test_chart_schemas_expose_only_relevant_inputs(self):
        self.assertNotIn('bins', scatter_plot.args)
        self.assertNotIn('x', correlation_heatmap.args)
        self.assertNotIn('y', histogram.args)
        self.assertIn('bins', histogram.args)

    def test_filtered_scatter_and_statistics_use_same_subset(self):
        filters = [{'column': 'age_years', 'op': 'gt', 'value': 30}]
        model = Model([call('calculate_statistics', {'operation': 'pearson', 'columns': ['heart_rate_bpm', 'sleep_hours'], 'filters': filters}),
                       call('scatter_plot', {'x': 'heart_rate_bpm', 'y': 'sleep_hours', 'filters': filters}, 'filtered-chart'),
                       answer('The correlation uses two complete pairs after filtering; one incomplete pair was excluded.')])
        result = self.run_graph(model, 'Show the relationship between heart rate and sleep hours above age 30.')['result']
        facts = result['evidence'][0]
        self.assertEqual(facts['filtered_out'], 2)
        self.assertEqual(facts['rows_used'], 2)
        self.assertEqual(facts['rows_excluded'], 1)
        self.assertEqual(len(result['charts'][0]['points']), 2)

    def test_overview_contract_requires_inspection_even_if_planner_flags_are_false(self):
        plan = {'intent': 'overview', 'goal': 'Introduce the dataset', 'steps': [], 'requires_calculation': False, 'requires_chart': False}
        with patch(__name__ + '.test_plan', return_value=plan):
            result = self.run_graph(Model([call('inspect_dataset', {}), answer('The table contains five records and four columns, with one missing sleep value.')]), 'Help me understand this dataset.')['result']
        self.assertFalse(result['partial'])
        self.assertEqual(result['evidence'][0]['operation'], 'overview')
        self.assertEqual(result['evidence'][0]['category_examples']['group'], ['A', 'B'])

    def test_repeated_overview_reuses_valid_dataset_inspection_before_calling_tools(self):
        def plan(state):
            return {'intent': 'overview', 'goal': 'Introduce the dataset', 'steps': ['inspect_dataset'], 'requires_calculation': True, 'requires_chart': False}
        with patch(__name__ + '.test_plan', side_effect=plan):
            self.run_graph(Model([call('inspect_dataset', {}), answer('The table contains five records.')]), 'Introduce this dataset.')
            model = Model([answer('The saved inspection has five records and one missing sleep value.')])
            state = self.run_graph(model, 'Help me understand this dataset again.')
        self.assertFalse(state['result']['partial'])
        self.assertNotIn('tools', state['trace'])
        self.assertEqual(state['result']['evidence'][0]['rows'], 5)
        self.assertTrue(state['reused_result_ids'])

    def test_planner_receives_conversation_as_task_data_not_a_chat_to_continue(self):
        question = 'Recall the earlier correlation; do not redo it.'
        state = {'question': question, 'profile': {'columns': []}, 'messages': [
            HumanMessage(content='Compare the measurements.'), AIMessage(content='The saved r is -0.096 with 166 pairs.'),
            HumanMessage(content=question)]}
        messages = planning_messages(state)
        self.assertEqual(len(messages), 2)
        self.assertIsInstance(messages[0], SystemMessage)
        self.assertIsInstance(messages[1], HumanMessage)
        payload = json.loads(messages[1].content)
        self.assertEqual(payload['new_question'], question)
        self.assertIn('do not answer', payload['task'])
        self.assertIn('166', json.dumps(payload['earlier_context']))

if __name__=='__main__': unittest.main()
