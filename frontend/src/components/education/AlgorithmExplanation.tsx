import { Card, CardContent, CardHeader, CardTitle } from "../ui/card"
import { Info } from "lucide-react"

interface AlgorithmExplanationProps {
    algorithmName: string
}

const ALGO_INFO: Record<string, {
    title: string
    type: string
    description: string
    formula: string
    pros: string[]
    cons: string[]
}> = {
    q_learning: {
        title: "Q-Learning",
        type: "Model-Free, Off-Policy, TD Control",
        description: "Q-Learning learns the value of the optimal policy independently of the agent's actions. It updates Q-values based on the maximum possible reward in the next state.",
        formula: "Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') - Q(s,a)]",
        pros: ["Guaranteed to converge to optimal policy", "Can learn from observation (off-policy)"],
        cons: ["Can overestimate values", "Maximization bias"]
    },
    sarsa: {
        title: "SARSA",
        type: "Model-Free, On-Policy, TD Control",
        description: "SARSA (State-Action-Reward-State-Action) learns the value of the policy being followed. It updates Q-values based on the actual action taken in the next state.",
        formula: "Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') - Q(s,a)]",
        pros: ["Safer learning (accounts for exploration)", "Converges for any policy"],
        cons: ["Slower convergence than Q-Learning", "Dependent on exploration strategy"]
    },
    monte_carlo: {
        title: "Monte Carlo",
        type: "Model-Free, On-Policy, MC Control",
        description: "Monte Carlo methods learn from complete episodes. They average the returns observed after visiting a state.",
        formula: "Q(s,a) ← average(Returns(s,a))",
        pros: ["Unbiased estimate of value", "No bootstrapping needed"],
        cons: ["High variance", "Requires episodic environments", "Updates only after episode ends"]
    },
    n_step_td: {
        title: "n-step TD",
        type: "Model-Free, On-Policy, TD Control",
        description: "n-step TD bridges the gap between one-step TD and Monte Carlo. It looks n steps ahead before bootstrapping.",
        formula: "G = r_t + γ r_{t+1} + ... + γ^n Q(s_{t+n}, a_{t+n})",
        pros: ["Faster learning than MC", "Lower variance than 1-step TD"],
        cons: ["More complex implementation", "Requires tuning 'n'"]
    },
    td_prediction: {
        title: "TD Prediction",
        type: "Model-Free, Policy Evaluation",
        description: "TD(0) Prediction estimates the value function V(s) for a given policy. It does not improve the policy (no control).",
        formula: "V(s) ← V(s) + α [r + γ V(s') - V(s)]",
        pros: ["Simple and fast", "Low variance"],
        cons: ["Prediction only (no control)", "Biased estimate"]
    },
    policy_iteration: {
        title: "Policy Iteration",
        type: "Model-Based, Dynamic Programming",
        description: "Policy Iteration alternates between evaluating the current policy and improving it until convergence. Requires a model of the environment.",
        formula: "V(s) = Σ p(s'|s,a) [r + γ V(s')]",
        pros: ["Converges to exact optimal policy", "Very fast convergence (few iterations)"],
        cons: ["Computationally expensive per iteration", "Requires known transition model"]
    },
    value_iteration: {
        title: "Value Iteration",
        type: "Model-Based, Dynamic Programming",
        description: "Value Iteration combines policy evaluation and improvement into a single step. It iteratively updates values based on the Bellman Optimality Equation.",
        formula: "V(s) ← max_a Σ p(s'|s,a) [r + γ V(s')]",
        pros: ["Simpler implementation than PI", "Converges to exact optimal policy"],
        cons: ["May be slower to converge than PI", "Requires known transition model"]
    }
}

export function AlgorithmExplanation({ algorithmName }: AlgorithmExplanationProps) {
    const info = ALGO_INFO[algorithmName]

    if (!info) return null

    return (
        <Card className="border-border bg-card/50 backdrop-blur-sm h-full">
            <CardHeader className="pb-2">
                <div className="flex items-center gap-2">
                    <Info className="h-5 w-5 text-blue-400" />
                    <CardTitle className="text-lg">{info.title}</CardTitle>
                </div>
                <p className="text-xs text-muted-foreground font-mono">{info.type}</p>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
                <p className="text-muted-foreground">{info.description}</p>

                <div className="bg-muted/50 p-3 rounded-md font-mono text-xs border border-border">
                    {info.formula}
                </div>

                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <h4 className="font-semibold mb-1 text-green-500 text-xs uppercase">Pros</h4>
                        <ul className="list-disc list-inside space-y-1 text-muted-foreground text-xs">
                            {info.pros.map((pro, i) => (
                                <li key={i}>{pro}</li>
                            ))}
                        </ul>
                    </div>
                    <div>
                        <h4 className="font-semibold mb-1 text-red-500 text-xs uppercase">Cons</h4>
                        <ul className="list-disc list-inside space-y-1 text-muted-foreground text-xs">
                            {info.cons.map((con, i) => (
                                <li key={i}>{con}</li>
                            ))}
                        </ul>
                    </div>
                </div>
            </CardContent>
        </Card>
    )
}
