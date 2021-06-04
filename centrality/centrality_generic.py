'''
closeness centrality evaluation 
for reference: 
https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.centrality.closeness_centrality.html
'''

import sys
#reload(sys)
#sys.setdefaultencoding('utf8')
import xlrd
import networkx as nx
import operator

G = nx.Graph()

ExcelFileName= 'Centrality of nodes -- adjacency matrix.xlsx'
workbook = xlrd.open_workbook(ExcelFileName)
worksheet = workbook.sheet_by_name("Sheet1")
num_rows = worksheet.nrows #Number of Rows
num_cols = worksheet.ncols #Number of Columns


result_data =[]
ordered_node_list = []
for curr_col in range(1, num_cols, 1):
	ordered_node_list.append(str(worksheet.cell_value(5, curr_col)))

################################Creating the network######################################
for curr_row in range(6, num_rows, 1):
	row_data = []
	for curr_col in range(0, num_cols, 1):
		data = str(worksheet.cell_value(curr_row, curr_col)) 
		row_data.append(data)
	for index in range(len(ordered_node_list)):
		if ordered_node_list[index] != row_data[0]:
			G.add_edge(row_data[0], ordered_node_list[index], weight = float(row_data[index+1]))

################################Storing the network#######################################			
print(nx.info(G))
fh=open("BlockChain.edgelist",'wb')
nx.write_edgelist(G, fh, data=True)

################################Evaluating Closeness Centrality###########################
scores = nx.closeness_centrality(G, distance = 'weight')
sorted_scores = sorted(list(scores.items()), key=operator.itemgetter(1), reverse = True)
print("city score")
for elm in sorted_scores:
    print(elm[0], elm[1])
#print(sorted_scores)
