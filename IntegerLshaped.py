from gurobipy import GRB
import gurobipy as gp
import copy
import pickle
import math
import random
import numpy as np
import time
import pdb
from abc import ABC, abstractmethod
from multiprocessing import Pool, Manager

class read_park_data:

    def __init__(self):
        self.candidate_dis = {}
        #self.exist_dis = []
        self.population = {}
        self.candidate = []
       # self.exist = []

    def read_data(self):

        f_read = open('candidate_dis.pkl','rb')
        self.candidate_dis=pickle.load(f_read)
        f_read.close()

        f_read = open('population.pkl', 'rb')
        self.population = pickle.load(f_read)
        f_read.close()

        f_read = open('candidate.pkl', 'rb')
        self.candidate = pickle.load(f_read)
        f_read.close()
class Two_Sp:

    def __init__(self, data_class):
        self.candidate=[]
        self.type = [0, 1, 2]
        self.cost={}
        self.population={}
        self.index_x=[]
        self.area=[50,100,200]
        #self.index_y=[]
        #self.index_q=[]
        #self.index_alpha = []
        #self.index_z = []
        #self.index_v = []
        # self.index_nt=[m for m in self.candidate]
        #self.index_kappa = []
        self.bigM=1000000000
        #self.beta_d={}
        #self.beta_type=20
        #self.beta_n=1
        self.x_dict={}
        self.mn_dict={}
        self.h_s=[]

        f_read = open('beta_type_scen.pkl', 'rb')
        beta_type = pickle.load(f_read)
        f_read.close()
        self.beta_type=beta_type[:5]

        f_read = open('beta_d_scen.pkl', 'rb')
        beta_d = pickle.load(f_read)
        f_read.close()
        self.beta_d=beta_d[:5]

        f_read = open('beta_n_scen.pkl', 'rb')
        beta_n = pickle.load(f_read)
        f_read.close()
        self.beta_n=beta_n[:5]
        #pdb.set_trace()



        self.candidate = data_class.candidate
        self.population = data_class.population
        self.candidate_dis = data_class.candidate_dis
        for j in range(len(self.candidate)):
            self.cost[(0, j)] = 10000
            self.cost[(1, j)] = 25000
            self.cost[(2, j)] = 60000

        self.index_x = [(i, m) for i in self.type for m in self.candidate]
        self.index_q = [(m, n) for m in self.candidate for n in self.population]
        self.index_alpha = [n for n in self.population]
        self.index_z = [(m, n) for m in self.candidate for n in self.population]
        self.index_y = [(i, m) for i in self.type for m in self.candidate]
        self.index_v = [(m, n) for m in self.candidate for n in self.population]
        #self.index_nt=[m for m in self.candidate]
        #self.index_kappa = [(i, m, n) for i in self.type for m in self.candidate for n in self.population]
        for i in range(len(self.index_x)):
            key=self.index_x[i]
            self.x_dict[key]=i

        for j in range(len(self.index_q)):
            key=self.index_q[j]
            self.mn_dict[key]=j

        self.scenario_index=list(range(len(self.beta_n)))
        self.n_scenario=len(self.scenario_index)

        self.n_candidate = len(self.candidate)
        self.n_population = len(self.population)
        self.n_type = len(self.type)

        h_s=[]
        for key in self.population.keys():
            h_s.append(self.population[key])
        h_s+=[0]*self.n_candidate*self.n_population*3
        h_s+=[-1]*self.n_population
        h_s+=[0]*self.n_candidate*self.n_population
        h_s+=[self.bigM]*self.n_candidate*self.n_population
        h_s+=[0]*self.n_candidate*self.n_population
        h_s+=[0]*self.n_candidate*self.n_type*2
        h_s+=[self.bigM]*self.n_candidate*self.n_type
        h_s+=[0]*self.n_candidate*self.n_population
        h_s+=[1]*self.n_candidate*self.n_population

        self.h_s=np.array(h_s)

        #count1 = 0
        #count2 = 0
        #count3 = 0
        #count4 = 0
        #for m in self.candidate:
        #    for n in self.population_scen[0].keys():
        #        if self.candidate_dis[m, n] < 0.5:
        #            self.beta_d[m, n] = 100
        #            #count1 += 1
        #        elif 0.5 <= self.candidate_dis[m, n] < 2.5:
        #            self.beta_d[m, n] = 50
        #            #count2 += 1
        #        elif 2.5 <= self.candidate_dis[m, n] < 5:
        #            self.beta_d[m, n] = 30
        #            #count3 += 1
        #        else:
        #            self.beta_d[m, n] = 10
                    #count4 += 1


    def get_main_lp(self):

        main_problem = gp.Model()


        x = main_problem.addVars(self.index_x, vtype="C", lb=0, ub=1, name='x')
        theta = main_problem.addVar(name="theta", vtype="C", lb=-10000000, ub=1000000000)
        #tau= main_problem.addVar(name="tau", vtype="C", lb=0)

        main_problem.setObjective(-theta,GRB.MINIMIZE)
        main_problem.addConstrs(gp.quicksum(x[(i,m)] for i in self.type)<= 1 for m in self.candidate)
        main_problem.addConstr(gp.quicksum(x[(i,m)]for i in self.type for m in self.candidate)>=10)
        main_problem.addConstr(gp.quicksum(self.cost[(i, m)] * x[(i, m)] for i in self.type for m in self.candidate)<= 2300000)

        return main_problem, x, theta


    def make_second_stage_model(self, scenario):
        """ Initializes a second stage model. """
        model = gp.Model()

        #population = self.population_scen[scenario]
        beta_type=self.beta_type[scenario]
        beta_d=self.beta_d[scenario]
        beta_n=self.beta_n[scenario]

        #index_q = [(m, n) for m in self.candidate for n in population]
        #index_alpha = [n for n in population]
        #index_z = [(m, n) for m in self.candidate for n in population]
        #index_v = [(m, n) for m in self.candidate for n in population]
        # self.index_nt=[m for m in self.candidate]
        #index_kappa = [(i, m, n) for i in self.type for m in self.candidate for n in population]

        #var_x={}

        #for (i, m) in self.index_x:
        #    v_name = f"x_{(i,m)}"
        #    var_x[(i,m)]=model.addVar(name=v_name, vtype="B")

        var_x=model.addVars(self.index_x, vtype=GRB.BINARY, name="x")
        var_q = model.addVars(self.index_q, vtype=GRB.CONTINUOUS, lb=0, name="q")
        var_alpha = model.addVars(self.index_alpha, vtype=GRB.CONTINUOUS, lb=0, name="alpha")
        var_z = model.addVars(self.index_z, vtype=GRB.BINARY, name="z")
        var_y = model.addVars(self.index_y, vtype=GRB.CONTINUOUS, lb=0, name="y")
        var_v = model.addVars(self.index_v, vtype=GRB.CONTINUOUS, lb=-10000000, ub=100000000, name="v")
        # self.var_nt = self.model.addVars(self.index_nt, vtype=GRB.CONTINUOUS, lb=0)
        #var_kappa = model.addVars(index_kappa, vtype=GRB.CONTINUOUS, lb=0)
        var_vmin=model.addVar(vtype=GRB.CONTINUOUS, lb=-10000000,ub=100000000, name="v_min")
        #var_u=model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="u")
        #var_tau=model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="tau")

        model.setObjective(-var_vmin,GRB.MINIMIZE)

        #model.addConstr(var_u>=var_tau-var_vmin)
        model.addConstrs(gp.quicksum(var_q[(m, n)] for m in self.candidate) <= self.population[n] for n in
                         self.population.keys())
        model.addConstrs(
            gp.quicksum(var_x[(i, m)] for i in self.type) >= var_z[(m, n)] for m in self.candidate for n in self.population.keys())
        model.addConstrs(
            var_q[(m, n)] <= self.bigM * var_z[(m, n)] for m in self.candidate for n in self.population.keys())
        model.addConstrs(
            self.bigM * var_q[(m, n)] >= var_z[(m, n)] for m in self.candidate for n in self.population)
        model.addConstrs(gp.quicksum(var_z[(m, n)] for m in self.candidate) >= 1 for n in self.population)
        model.addConstrs(var_v[(m,n)]==gp.quicksum(var_x[(i,m)] for i in self.type)*beta_d[(m,n)]+
                              gp.quicksum(var_x[(i,m)]*beta_type[i] for i in self.type)-beta_n*gp.quicksum(var_y[(i,m)] for i in self.type)
                              for m in self.candidate for n in self.population)
        model.addConstrs(var_v[(m, n)] - var_alpha[n] >= self.bigM * (var_z[(m, n)] - 1)
                         for m in self.candidate for n in self.population)
        model.addConstrs(var_v[(m, n)] - var_alpha[n] <= 0 for m in self.candidate for n in self.population)
        model.addConstrs(
            var_y[(i, m)] <= self.bigM * var_x[(i, m)] for m in self.candidate for i in self.type)
        model.addConstrs(
            var_y[(i, m)] <= gp.quicksum(var_q[(m, n)] for n in self.population) * (1 - math.exp(-2)) /
            self.area[i] for i in self.type for m in self.candidate)
        model.addConstrs(var_y[(i, m)] >= gp.quicksum(var_q[(m, n)] for n in self.population)
                         * (1 - math.exp(-2)) / self.area[i] - self.bigM * (1 - var_x[(i, m)])
                         for i in self.type for m in self.candidate)
        model.addConstrs(var_vmin <= var_v[(m, n)] for m in self.candidate for n in self.population)


        model.update()

        return model

    def evaluate_first_stage_sol(self, x_dict, gap=0.001, time_limit=3600, threads=1, verbose=0):
        """ Gets the objective function value for a given solution. """
        #scenarios = self.scenario_index
        #n_scenarios = len(scenarios)
        scenario_prob = 1 / self.n_scenario

        # get first-stage objective
        #first_stage_obj_val = np.dot(x, self.fixed_costs)

        # get second-stage objective
        #pool = Pool(n_procs)
        #results = [pool.apply_async(self.get_second_stage_obj_worker,
        #                            args=(x, scenario, scenario_prob, gap, time_limit, threads, verbose)) for scenario
        #           in scenarios]
        #results = [r.get() for r in results]
        #pool.close()
        #pool.join()
        results=[]
        v_min_list=[]
        for scenario in self.scenario_index:
            theta_obj, ss_obj=self.get_second_stage_obj_worker( x_dict, scenario, scenario_prob, gap, time_limit, threads, verbose)
            results.append(theta_obj)
            v_min_list.append(ss_obj)

        second_stage_obj_val = np.sum(results)

        return  second_stage_obj_val, v_min_list

    def get_second_stage_obj_worker(self, x_dict, scenario, scenario_prob, gap, time_limit, threads, verbose):
        """ Multiprocessing for getting second-stage objective. """
        ss_obj=self.get_second_stage_objective(x_dict, scenario, gap=gap, time_limit=time_limit, threads=threads, verbose=verbose)
        second_stage_obj = scenario_prob * ss_obj
        return second_stage_obj, ss_obj

    def get_second_stage_objective(self, x_dict, scenario, gap=0.001, time_limit=36000, threads=1, verbose=0):
        """ Gets the second stage model for an objective. """
        model = self.make_second_stage_model(scenario)
        model = self.set_first_stage(model, x_dict)

        # optimize model
        model.setParam("OutputFlag", verbose)
        model.setParam("MIPGap", gap)
        model.setParam("TimeLimit", time_limit)
        model.setParam('Threads', threads)
        model.optimize()

        var = model.getVarByName("v_min")
        ss_obj = var.x

        #for var in model.getVars():
        #    if "v_min" in var.varName:
        #        ss_obj=var.x

        return ss_obj

    def set_first_stage(self, model, x_dict):
        """ Fixes the first stage solution of a given model. """
        for i in two_sp.index_x:
            var_name="x[{},{}]".format(i[0],i[1])
            var=model.getVarByName(var_name)
            #pdb.set_trace()
            var.ub=x_dict[var_name]
            var.lb=x_dict[var_name]
        #for var in model.getVars():
        #    if "x_" in var.varName:
        #        idx = int(var.varName.split("_")[-1])
        #        var.ub = x[idx-1]
        #        var.lb = x[idx-1]
        model.update()
        return model

    def get_second_stage_cost(self, model):
        """ Gets the second stage cost of a given model.  """
        ss_obj = 0
        var=model.getVarByName("v_min")
        #for var in model.getVars():
        #    if "y" in var.varName or "r" in var.varName:
        #        ss_obj += var.obj * var.x
        ss_obj=var.x
        return ss_obj

class IntegerLShaped(ABC):

    def __init__(self, two_sp, scenarios):

        self.two_sp = two_sp
        self.scenarios = scenarios
        self.n_scenarios = len(scenarios)
        self.benders_max_iter = 1000
        self.bender_tol=1e-6
        self.n_candidate=len(two_sp.candidate)
        self.n_population=len(two_sp.population)
        self.n_type=len(two_sp.type)
        self.big_m=1000000000



    def get_subproblems(self, as_lp):
        subproblems = []
        for scen in self.scenarios:
            Q_s=self.two_sp.make_second_stage_model(scen)
            #pdb.set_trace()
            if as_lp:
                Q_s=self.fix_second_stage_model(Q_s,as_lp)

            Q_s.setParam("OutputFlag", 0)
            Q_s.setParam("Threads", 1)
            Q_s.setParam("MipGap", 0.001)

            Q_s._pi_constrs = Q_s.getConstrs()
            x_list=[]
            for i in self.two_sp.index_x:
                vname="x[{},{}]".format(i[0],i[1])
                x_list.append(Q_s.getVarByName(vname))
            Q_s._x=x_list

            subproblems.append(Q_s)
        return subproblems

    def fix_second_stage_model(self, Q_s, as_lp):

        if as_lp:
            # relax mip to LP
            Q_s = Q_s.relax()

            # add constraints for upper bound of y values
            for var in Q_s.getVars():
                if "z" in var.varName:
                    var.ub = gp.GRB.INFINITY
                    Q_s.addConstr(var + 0 <= 1, name="{var.varName}_ub")

        Q_s.update()

        return Q_s

    def get_second_stage_info(self, scen_index):
        n_constraints=self.n_population*2+self.n_candidate*self.n_population*8\
                      +self.n_candidate*self.n_type*3
        beta_type = two_sp.beta_type[scen_index]
        beta_d = two_sp.beta_d[scen_index]
        #beta_n = two_sp.beta_n[scen_index]
        T_s=np.zeros((n_constraints,self.n_candidate*self.n_type))
        for key in two_sp.mn_dict.keys():
            for i in two_sp.type:
                T_s[two_sp.mn_dict[key]+self.n_population,two_sp.x_dict[(i,key[0])]]=-1
                col1=self.n_population*2+self.n_candidate*self.n_population*3
                T_s[two_sp.mn_dict[key]+col1, two_sp.x_dict[(i,key[0])]]=beta_d[key]+beta_type[i]

        for key in two_sp.x_dict:
            col2 = self.n_population * 2 + self.n_candidate * self.n_population * 6
            T_s[two_sp.x_dict[key]+col2, two_sp.x_dict[key]]=-self.big_m
            col3=self.n_population * 2 + self.n_candidate * self.n_population * 6+self.n_candidate*self.n_type*2
            T_s[two_sp.x_dict[key]+col3, two_sp.x_dict[key]]=self.big_m

        h_s=two_sp.h_s
        #for key in two_sp.population.keys():
        #    h_s.append(two_sp.population[key])
        #h_s+=[0]*self.n_candidate*self.n_population*3
        #h_s+=[-1]*self.n_population
        #h_s+=[self.big_m]*self.n_candidate*self.n_population
        #h_s+=[0]*self.n_candidate*self.n_population
        #h_s+=[0]*self.n_candidate*self.n_type*2
        #h_s+=[self.big_m]*self.n_candidate*self.n_type
        #h_s+=[1]*self.n_candidate*self.n_population


        return T_s, h_s


    #def get_lower_bound_first_stage(self):



    def compute_lower_bound(self, as_lp, main_problem):

        #x={}

        #for i in self.two_sp.index_x:
        #    x[i]=0

        results = []

        #p_s=1/self.n_scenarios

        for scen_idx in self.scenarios:
            results.append(0)
            #results.append(self.compute_lb_worker(x, p_s, scen_idx, as_lp))

        lower_bound = np.sum(results)

        return lower_bound

    def compute_lb_worker(self, x, p_s, scen_idx, as_lp):

        if as_lp:
            Q_s = main_problem._lp_subproblems[scen_idx]
        else:
            Q_s = main_problem._ip_subproblems[scen_idx]

        Q_s = self.fix_first_stage_sol(Q_s, x)
        Q_s.optimize()

        return p_s * Q_s.objVal

    def fix_first_stage_sol(self, Q_s, x):
        """ Fixes first-stage decision in second-stage model.

                    Parameters
                    -------------------------------------------------
                        Q_s - A gurobi model of the second-stage problem
                        x - The solution to set the first-stage decision to
        """

        Q_s.setAttr(gp.GRB.Attr.LB, Q_s._x, x)
        Q_s.setAttr(gp.GRB.Attr.UB, Q_s._x, x)
        #self.lower_bound = lower_bound
        return Q_s

    def set_lower_bound(self, main_problem, lower_bound):
        main_problem._lower_bound = lower_bound
        main_problem._theta.lb = lower_bound
        self.lower_bound = lower_bound

        return main_problem
    def benders(self, main_problem):
        """ Benders decompostion for linear relaxation of master problem. """
        time_benders = time.time()

        for it in range(self.benders_max_iter):

            if (it % 10) == 0:
                print(f"    Iteration: {it}")

            # optimize main problem
            main_problem.optimize()

            # recover solution
            x = list(map(lambda x: x.x, main_problem._x.values()))
            theta = main_problem._theta.x

            sp_time = time.time()

            results = []

            for scen_idx in self.scenarios:
                p_s=1/self.n_scenarios
                T_s, h_s=self.get_second_stage_info(scen_idx)

                results.append(self.get_sp_cut_worker(x, p_s, h_s, T_s, scen_idx, main_problem))

            sp_time = time.time() - sp_time

            cut_time = time.time()

            Q = np.sum(list(map(lambda x: x[0], results)))
            a_sum = np.sum(list(map(lambda x: x[1], results)))
            b_sum = np.sum(list(map(lambda x: x[2], results)), axis=0)

            if theta >= Q - self.bender_tol:
                print("    Done: theta >= Q")
                break

            # otherwise add cut
            x_beta_sum_ = 0
            for i, x_var in main_problem._x.items():
                x_beta_sum_ += b_sum[i] * x_var

            main_problem.addConstr(main_problem._theta >= a_sum - x_beta_sum_, name=f"bc_{main_problem._bd_cuts}")
            main_problem._bd_cuts += 1

        time_benders = time.time() - time_benders
        main_problem._time_benders = time_benders

    def get_sp_cut_worker(self, x, p_s, h_s, T_s, scen_idx, main_problem):
        """ Function for get sp cuts.
                    Used in benders as well as the relaxed cuts in integer L-shaped method.

                    Parameters
                    -------------------------------------------------
                        x: the first-stage decision
                        p_s: a float of the scenario probability
                        h_s: a list or array of the rhs for a scenario
                        T_s: a (nested) list or array of the linking constraints for a scenario.
                        scen_idx:  the index of the scenario
                        as_lp: an indicator for getting subproblems with or without the LP relaxation.
        """

        # fix first-stage variables
        fix_time = time.time()
        #pdb.set_trace()
        Q_s = main_problem._lp_subproblems[scen_idx]
        Q_s = self.fix_first_stage_sol(Q_s, x)
        Q_s.setParam("Threads", 1)
        Q_s.setParam("OutputFlag", 1)
        fix_time = time.time() - fix_time

        # optimize
        opt_time = time.time()
        Q_s.optimize()
        opt_time = time.time() - opt_time
        #pdb.set_trace()

        # get weighted objective
        q_val = p_s * Q_s.objVal

        # recover dual value
        dual_time = time.time()
        pi=Q_s.getAttr(gp.GRB.Attr.Pi, Q_s._pi_constrs)
        dual_time = time.time() - dual_time

        # compute alpha/beta for cuts
        cut_time = time.time()
        alpha = np.multiply(p_s, np.dot(pi, h_s))
        beta = np.multiply(p_s, np.matmul(pi, T_s))
        cut_time = time.time() - cut_time

        info = {
            "fix_time": fix_time,
            "opt_time": opt_time,
            "dual_time": dual_time,
            "cut_time": cut_time,
        }

        # check that values are correct
        assert (np.abs(alpha - np.dot(beta, x) - q_val) < 1e-2)

        return q_val, alpha, beta, info

    def add_integer_optimality_cut(self, main_problem, x, Q):
        """ Add integer optimality cut to main_problem.

                    Parameters
                    -------------------------------------------------
                        x: the first-stage decision in callback
                        Q: the value for  the integer cut
                """
        x = list(map(lambda y: y, x.values()))
        x = list(map(lambda x: max(x, 0), x))  # fix value to be >= 0
        x = list(map(lambda x: min(x, 1), x))  # fix value to be <= 1

        # compute the set S for integer cut
        S = []
        S_not = []
        for i, var in main_problem._x.items():
            if x[i] > 0.99:
                S.append(var)
            else:
                S_not.append(var)

        # compute and add integer cut
        integer_cut = (Q - self.lower_bound) * (sum(S) - sum(S_not) - len(S)) + Q
        main_problem.cbLazy(main_problem._theta >= integer_cut)
        main_problem._io_cuts += 1

        return main_problem

    def add_subgradient_cut(self, main_problem, alpha, beta):
        x_beta_sum_ = 0
        for i, x_var in main_problem._x.items():
            x_beta_sum_ += beta[i] * x_var
        main_problem.cbLazy(main_problem._theta >= alpha - x_beta_sum_)
        main_problem._sg_cuts += 1
        return main_problem

    def hash_x(self, x):
        """ Hashes binary first-stage solution to tuple.  Rounds values to avoid numerical issues.

            Parameters
            -------------------------------------------------
                x - the first-stage decision (as a list).
        """
        xh = x.select()
        for i in range(len(xh)):
            if xh[i] < 0.5:
                xh[i] = 0
            else:
                xh[i] = 1
        return tuple(xh)

def alternating_cuts(main_problem, where):
    """ Callback function for integer L-shaped method with alternating cuts.  """
    if where == gp.GRB.Callback.MIPSOL:

    # get number of nodes
        n_nodes = main_problem.cbGet(gp.GRB.Callback.MIPSOL_NODCNT)
        if n_nodes % 10 == 0:
            print("  # Nodes:", n_nodes)

    # get first-stage and theta
        x = main_problem.cbGetSolution(main_problem._x)
        theta = main_problem.cbGetSolution(main_problem._theta)


        if ils.hash_x(x) in main_problem._V:
            print("  Solution (x) in V, ending callback")
            return

        if ils.hash_x(x) not in main_problem._V_lp:
            print("  Solution (x) not in V_lp...")

            # compute Q_lp, and subgradient cut info
            Q_lp, alpha, beta = compute_subgradient_cut(x, theta)

            # add x to V_LP
            main_problem._V_lp.add(ils.hash_x(x))

            # add sg cut and return if theta < Q_lp
            if theta < Q_lp:
                print("  Adding subgradient cut (theta < Q_lp), ending callback")
                main_problem = ils.add_subgradient_cut(main_problem, alpha, beta)
                return

            print("  No subgradient cut needed (theta >= Q_lp)")

        # integer optimality cut Q value
        Q_ip = compute_integer_optimality_cut(x, theta)

        # if sg cuts, then add integer cuts
        main_problem._V.add(ils.hash_x(x))

        if theta < Q_ip:
            print("  Adding optimality (theta < Q_ip), ending callback")
            main_problem = ils.add_integer_optimality_cut(main_problem, x, Q_ip)
            return

        print("Did not meet any conditions!")


def compute_subgradient_cut(x, theta):
    """ Computes subgradient cuts based on IP solution of model.

                Parameters
                -------------------------------------------------
                    x: the first-stage decision in callback
                    theta: the variable theta in callback
    """
    # x to list of values
    x = list(map(lambda y: y, x.values()))
    x = list(map(lambda x: max(x, 0), x))  # fix value to be >= 0
    x = list(map(lambda x: min(x, 1), x))  # fix value to be <= 1

    sp_time = time.time()
    results = []

    p_s = 1 / ils.n_scenarios
    for scen_idx in range(ils.n_scenarios):
        T_s, h_s=ils.get_second_stage_info(scen_idx)

        # compute sp cut info
        print(f"     Subgradient cut for scen {scen_idx}/{ils.n_scenarios}")
        results.append(ils.get_sp_cut_worker(x, p_s, h_s, T_s, scen_idx))

    sp_time = time.time() - sp_time

    Q = np.sum(list(map(lambda x: x[0], results)))
    alpha = np.sum(list(map(lambda x: x[1], results)))
    beta = np.sum(list(map(lambda x: x[2], results)), axis=0)

    return Q, alpha, beta




def compute_integer_optimality_cut(x, theta):
    # x to list of values
    x = list(map(lambda y: y, x.values()))
    x = list(map(lambda x: max(x, 0), x))  # fix value to be >= 0
    x = list(map(lambda x: min(x, 1), x))  # fix value to be <= 1

    sp_time = time.time()
    results = []

    p_s = 1 / ils.n_scenarios
    for scen_idx in ils.scenarios:
        # get relavent constraint/coefficient info


        # print(f"     Optimality cut for scen {scen_idx}/{ils.n_scenarios}")
        results.append(get_optimality_cut_worker(x, p_s, scen_idx))

    sp_time = time.time() - sp_time

    Q = np.sum(results)

    return Q


def get_optimality_cut_worker(x, p_s, scen_idx):
    """ Multiprocessing function for get sp cut.

        Parameters
        -------------------------------------------------
            x: the first-stage decision
            p_s: a float of the scenario probability
            scen_idx:  the index of the scenario
    """
    # load subproblem, fix x, then optimize
    Q_s = main_problem._ip_subproblems[scen_idx]
    Q_s = ils.fix_first_stage_sol(Q_s, x)
    Q_s.optimize()

    q_val = p_s * Q_s.objVal

    return q_val




if __name__ == "__main__":

    data = read_park_data()
    data.read_data()
    two_sp=Two_Sp(data)

    global main_problem
    main_problem, x, theta = two_sp.get_main_lp()
    scenarios=two_sp.scenario_index

    main_problem._x = x
    main_problem._theta = theta

    main_problem.setParam("MipGap", 0.001)
    main_problem.setParam("Threads", 1)
    main_problem.setParam("OutputFlag", 0)

    global ils
    ils = IntegerLShaped(two_sp, scenarios)

    lp_subproblems = ils.get_subproblems(as_lp=True)
    main_problem._lp_subproblems = lp_subproblems

    # get integer subproblems
    ip_subproblems = ils.get_subproblems(as_lp=False)
    main_problem._ip_subproblems = ip_subproblems


    #ils.get_second_stage_info()

    print("\nComputing Lower Bound...")
    time_lower_bound = time.time()

    #lower_bound = ils.compute_lower_bound(main_problem)
    lower_bound=0
    main_problem=ils.set_lower_bound(main_problem, lower_bound)

    time_lower_bound = time.time() - time_lower_bound

    print("  Lower bound compute:", lower_bound)
    print("  Time for lower bound:", time_lower_bound)

    print("\nBenders Decomposition...")

    main_problem._bd_cuts = 0
    main_problem._sg_cuts = 0
    main_problem._io_cuts = 0
    main_problem.Params.lazyConstraints = 1

    main_problem._V = set()
    main_problem._V_lp = set()

    ils.benders(main_problem)

    benders_obj = main_problem.objVal

    print("  Objective of LP Relaxation:                ", benders_obj)
    print("  Solving time for Benders (LP Relaxation):  ", main_problem._time_benders)

    # integer L-shaped method
    print("\nInteger L-shaped Method...")
    for _, var in main_problem._x.items():
        var.vtype = "B"

    main_problem.optimize(alternating_cuts)

    print("\n\nSummary:")
    print("  Bender's cuts:", main_problem._bd_cuts)
    print("  Subgradient cuts:", main_problem._sg_cuts)
    print("  Integer cuts:", main_problem._io_cuts)

    print("  Number of Nodes:", main_problem.NodeCount)

    print("  Benders time:", main_problem._time_benders)
    print("  Time ILS:", main_problem.RunTime)
    print("  Time total:", main_problem._time_benders + main_problem.RunTime)

    # get first-stage decision
    x = main_problem._x.select()
    x_dict={}
    #for i in two_sp.index_x:
    #    varname="x[{},{}]".format(i[0], i[1])
    #pdb.set_trace()

    for i in range(len(x)):
        varname=x[i].VarName
        x_dict[varname]=x[i].x

    #x = list(map(lambda y: y.x, x))

    # get objective from first-stage decision
    fs_obj,obj_list = two_sp.evaluate_first_stage_sol(x_dict)

    print("\n  First-stage decision:", x)
    print("  First-stage solution obj:", fs_obj)

    # collect and store all results
    results = {
        'time_lower_bound': time_lower_bound,
        'time_benders': main_problem._time_benders,
        'time_integer_l_shaped': main_problem.RunTime,
        'time_total': time_lower_bound + main_problem._time_benders + main_problem.RunTime,

        'obj_benders': benders_obj,
        'obj_fs': fs_obj,

        'cuts_benders': main_problem._bd_cuts,
        'cuts_subgradient': main_problem._sg_cuts,
        'cuts_integer_opt': main_problem._io_cuts,

        'n_nodes': main_problem.NodeCount,

        'x': x,
    }

    #problem_str = f"s{args.n_scenarios}"
    #fp_results = get_path(args.data_path, args.problem, ptype=f"ils_{problem_str}", suffix=".pkl")

    #with open(fp_results, 'wb') as p:
        #pkl.dump(results, p)
    x = list(map(lambda y: y.x, x))
    abc=sum(x)
    print(abc)