import { useMemo } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card"
import { Compass, ArrowUp, ArrowRight, ArrowDown, ArrowLeft } from "lucide-react"

interface PolicyViewProps {
    envName: string
    policy: Record<string, number>
}

export function PolicyView({ envName, policy }: PolicyViewProps) {
    const gridSize = useMemo(() => {
        if (envName === "gridworld") return 5
        if (envName === "frozenlake") return 4
        return 0
    }, [envName])

    const getArrow = (action: number) => {
        // GridWorld: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        if (envName === "gridworld") {
            switch (action) {
                case 0: return <ArrowUp className="h-4 w-4" />
                case 1: return <ArrowRight className="h-4 w-4" />
                case 2: return <ArrowDown className="h-4 w-4" />
                case 3: return <ArrowLeft className="h-4 w-4" />
            }
        }
        // FrozenLake: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP
        if (envName === "frozenlake") {
            switch (action) {
                case 0: return <ArrowLeft className="h-4 w-4" />
                case 1: return <ArrowDown className="h-4 w-4" />
                case 2: return <ArrowRight className="h-4 w-4" />
                case 3: return <ArrowUp className="h-4 w-4" />
            }
        }
        return null
    }

    if (gridSize === 0) {
        return (
            <Card className="border-border bg-card/50 backdrop-blur-sm h-full">
                <CardHeader>
                    <CardTitle className="text-sm text-muted-foreground">
                        Policy visualization not available for this environment.
                    </CardTitle>
                </CardHeader>
            </Card>
        )
    }

    const cells = Array.from({ length: gridSize * gridSize }, (_, i) => i)

    return (
        <Card className="border-border bg-card/50 backdrop-blur-sm h-full">
            <CardHeader className="pb-2">
                <div className="flex items-center gap-2">
                    <Compass className="h-5 w-5 text-purple-500" />
                    <CardTitle className="text-lg">Learned Policy</CardTitle>
                </div>
            </CardHeader>
            <CardContent>
                <div
                    className="grid gap-1 w-full aspect-square max-w-[300px] mx-auto"
                    style={{
                        gridTemplateColumns: `repeat(${gridSize}, 1fr)`,
                    }}
                >
                    {cells.map((stateId) => {
                        const action = policy[stateId.toString()]
                        // If action is undefined, show nothing or a dot
                        if (action === undefined) {
                            return (
                                <div
                                    key={stateId}
                                    className="flex items-center justify-center border border-border/50 rounded-sm bg-muted/20"
                                >
                                    <span className="h-1 w-1 rounded-full bg-muted-foreground/50" />
                                </div>
                            )
                        }

                        return (
                            <div
                                key={stateId}
                                className="flex items-center justify-center border border-border/50 rounded-sm bg-muted/20 text-foreground/80 hover:bg-muted/40 transition-colors"
                                title={`State ${stateId}: Action ${action}`}
                            >
                                {getArrow(action)}
                            </div>
                        )
                    })}
                </div>
                <div className="text-center text-xs text-muted-foreground mt-4">
                    Arrows indicate the greedy action for each state.
                </div>
            </CardContent>
        </Card>
    )
}
