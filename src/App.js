import React from 'react'
import BarChart from './js/BarChart'
import TopicBarChart from './js/TopicBarChart'
import ScatterPlot from './js/ScatterPlot'
import DensityPlot from './js/DensityPlot'
import Lists from './js/Lists'
import TopicSelector from './js/TopicSelector'
import { Button, Dialog } from '@blueprintjs/core'
import SketchPicker from 'react-color'
import ta from './data/topic_files/topic_2.json'
import tb from './data/topic_files/topic_3.json'
// import Filter from './js/Filter'

export default class Main extends React.Component {
    constructor(props) {
        super(props);

        const topicA = 1; // change import ta above as well
        const topicB = 2; // change import tb above as well
        let topics = {
            [topicA]: ta,
            [topicB]: tb
        };
        // for (let i = 0; i < 50; i++) {
        // let t = require(`./data/topic_files/topic_${i + 1}.json`);
        // topics = [...topics, t];
        // }

        const topicScores = require('./data/topic_scores.json');
        const jokes = require('./data/jokefile.json');

        // initial app state
        this.state = {
            jokes,
            topics,
            topicA, // default topic A
            topicB, // default topic B
            topicScores,
            topicA_color: '#32973F',    // color of 1st topic
            topicB_color: '#236C92',    // color of 2nd topic
            words_to_show: 100,         // number of words to show in the bar chart
            enableDistortion: false,    // toggle cartesian distortion
            helpIsOpen: false,          // help dialog
            displayColorPicker: false,  // color picker
            color_to_change: -1,
            bar_selection: {
                open: false,
                topic: '',
                index: -1,
                word: null,
            },
            word_page: 0,
            words_per_bar: 21
        }
    }

    setTopicA = async (topicA) => {
        if (!this.state.topics[topicA]) {
            this.setState({
                topics: {
                    ...this.state.topics,
                    [topicA]: await import(`./data/topic_files/topic_${topicA + 1}.json`),
                },
                topicA

            });
        } else {
            this.setState({
                topicA
            });
        }
    }

    setTopicB = async (topicB) => {
        if (!this.state.topics[topicB]) {
            const data = await import(`./data/topic_files/topic_${topicB + 1}.json`);
            this.setState({
                topics: {
                    ...this.state.topics,
                    [topicB]: data
                },
                topicB
            });
        } else {
            this.setState({
                topicB
            });
        }
    }

    toggleHelpOverlay = () => {
        this.setState({
            helpIsOpen: !this.state.helpIsOpen
        })
    }

    toggleDistortion = () => {
        this.setState({
            enableDistortion: !this.state.enableDistortion
        })
    }

    toggleColorPicker = (t) => {
        this.setState({
            color_to_change: t,
            displayColorPicker: !this.state.displayColorPicker
        });
    }

    handleColorChange = (color) => {
        if (this.state.color_to_change === 1) {
            this.setState({
                topicA_color: color.hex,
            })
        }
        else {
            this.setState({
                topicB_color: color.hex,
            })
        }
    }

    setTopicAWord = (w, i) => {
        this.setState({
            bar_selection: {
                open: true,
                topic: 'topicA',
                index: i,
                word: w
            }
        })
    }

    setTopicBWord = (w, i) => {
        this.setState({
            bar_selection: {
                open: true,
                topic: 'topicB',
                index: i,
                word: w
            }
        })
    }

    clearSelection = () => {
        this.setState({
            bar_selection: {
                open: false,
                topic: '',
                index: -1,
                word: null,
            }
        });
    }

    // removed per Brock's suggestion
    // setWordPage = (s) => {
    //     this.setState({
    //         word_page: s,
    //     })
    // }

    switchActiveView = (content) => {
        var toggle = (el, s) => {

            if (el) {
                if (s === "hide" && !el.classList.contains("hidden"))
                    el.className += " hidden"

                else if (s === "show" && el.classList.contains("hidden"))
                    el.classList.remove("hidden")                    
            }
        }

        var toggleScatter = (s) => {
            var el =document.getElementById("scatter_plot")
            toggle(el, s)
        }

        var toggleBars = (s) => {
            var el = document.getElementById("bar_chart_container")
            toggle(el, s)  
        }

        var toggleTopicChart = (s) => {
            var el = document.getElementById("topic_chart")
           toggle(el, s)
        }

        var toggleDensity = (s) => {
            var el = document.getElementById("density_plot")
            toggle(el, s)
        }

        if (content === "scatter" ) {
            toggleScatter("show");

            toggleBars("hide");  
            toggleTopicChart("hide");  
            toggleDensity("hide");
        }

        else if (content === "bars") {
            toggleBars("show")

            toggleScatter("hide");
            toggleTopicChart("hide");
            toggleDensity("hide");
        }

        else if(content === "density") {
            toggleDensity("show");

            toggleScatter("hide");
            toggleTopicChart("hide");
            toggleBars("hide");
        }

        else if(content === "topics") {
            toggleTopicChart("show");
            
            toggleScatter("hide");
            toggleDensity("hide");
            toggleBars("hide");
        }

        else if(content === "all") {
            toggleDensity("show");
            toggleScatter("show");
            toggleTopicChart("show");
            toggleBars("show");
        }
    }

    render() {
        return (
            <div>
                <nav id="interactions">
                    <Button text={'D'}
                        onClick={this.toggleDistortion}
                        className={"small float_left"} />

                    <TopicSelector
                        position={1}
                        value={this.state.topicA}
                        onChange={this.setTopicA} />

                    <Button
                        className={"selector color_selector float_left"}
                        onClick={this.toggleColorPicker.bind(this, 1)}
                        style={{ background: this.state.topicA_color }} />

                    <a href="https://www.reddit.com/r/jokes/">
                        <img id="logo" src={require("./css/reddit_logo.png")} alt="reddit logo" />
                    </a>

                    <Button text="?"
                        onClick={this.toggleHelpOverlay}
                        className={"small float_right"} />

                    <TopicSelector
                        position={2}
                        value={this.state.topicB}
                        onChange={this.setTopicB} />

                    <Button
                        id={"topic_b_color_selector"}
                        className={"selector color_selector float_right"}
                        onClick={this.toggleColorPicker.bind(this, 2)}
                        style={{ background: this.state.topicB_color }} />
                </nav>

                <nav id="page_selection_nav">
                 <div className="page_selector" onClick={this.switchActiveView.bind(this, "scatter")} >
                    <span>See the relative frequency of words in these topics</span>
                 </div>
                 <div className="page_selector" onClick={this.switchActiveView.bind(this, "bars")}>
                    <span>See which words make up the topic</span>
                 </div>
                 <div className="page_selector" onClick={this.switchActiveView.bind(this, "density")}>
                    <span>See how well these topics scored</span>
                 </div>
                 <div className="page_selector" onClick={this.switchActiveView.bind(this, "topics")}>
                    <span>See how all topics scored</span>
                 </div>
                 <div className="page_selector" onClick={this.switchActiveView.bind(this, "all")}>
                    <span>See them all together</span>
                 </div>
                </nav>

                {/* help dialog */}
                <Dialog
                    isOpen={this.state.helpIsOpen}
                    onClose={this.toggleHelpOverlay}
                    hasBackdrop={true}
                    title={"Help"} >
                    <div className="help_content">
                        <h6>What are we looking at?</h6>
                        <p>This web app visualizes the results of performing LDA topic modeling analysis with the Reddit joke dataset - 200K jokes!</p>

                        <h6>Switching Topics</h6>
                        <p>Topics can be changed with the dropdowns in the upper left and right of the navbar. Their associated <strong>colors</strong> can also be changed by clicking on the color squares next to the dropdowns!</p>

                        <h6>Scatter Plot</h6>
                        <p>Each dot on the scatter plot represents a word. The dot's position along the horizontal and vertical axises describe the probablity that the associated word belongs to the first or second selected topics respectively. The hue of the dot helps distinguish associated topics as well indicating how probable the associated word belongs to a particular topic. The stronger the hue is towards a topic's main color the more probable they're associated!</p>
                        <p><strong>Cartesian Distortion</strong> is a means to scale the axese on demand allowing you to add separation inbetween the closely clustered words. It can be applied to the scatter plot to help compare word positions. You can enable or disable it by clicking the 'D' in the far left of the nav bar.</p>

                        <h6>Bar Plots</h6>
                        <p>Most text modeling algorithms tend to destroy the semantic context associated with their results. Unfortunately, LDA modeling is one of them. As such, we're only able to discover a predetermined number of topics but not give a topic name to bring context to the associated words. Our model performed the best when tuned to 50 topics. The bar plots are used to give you back some more of that semantic context. It's quite fun to flip through the topics and see what they're about. A longer bar indicates higher word to topic affinity and a brighter hue indicates higher word count.</p>

                        <h6>Area (Density) Plot</h6>
                        <p>The area chart shows how likely a joke written about a particular topic is going to recieve more views. If the area is all crammed up towards the left don't write a joke in that topic because it's unlikely to score well!</p>
                    </div>
                </Dialog>

                {/* color picker */}
                < Dialog
                    isOpen={this.state.displayColorPicker}
                    onClose={this.toggleColorPicker}
                    className={"color_container"} >
                    <SketchPicker
                        onChangeComplete={this.handleColorChange}
                        color={
                            this.state.color_to_change === 1 ?
                                this.state.topicA_color :
                                this.state.topicB_color}
                        className={"colorSelector"} />
                </Dialog>

                <div>
                    <section id="scatter_plot">
                        <ScatterPlot {...this.state} />
                    </section>

                    <div id="density_plot">
                        <DensityPlot {...this.state} />
                    </div>

                    <div id="bar_chart_container">
                        <div className="bar_charts">
                            <BarChart
                                {...this.state}
                                onSelect={this.setTopicBWord}
                                topic={2}
                                page={this.word_page} />
                        </div>
                        <div className="bar_charts">
                            <BarChart
                                {...this.state}
                                onSelect={this.setTopicAWord}
                                topic={1}
                                page={this.word_page} />
                        </div>
                    </div>
                    
                    {/*   <div id="bar_char_nav">
                        <Button
                            className={"page_button flip"}
                            onClick={this.setWordPage.bind(this, 0)}>
                            <img className="paging_button" src={require("./css/chev_dbl.png")} alt="page right" />
                        </Button>
                        <Button
                            className={"page_button flip"}
                            disabled={this.state.word_page === 0 ? true : false}
                            onClick={this.setWordPage.bind(this, this.state.word_page - 1)}>
                            <img className="paging_button" src={require("./css/chev.png")} alt="page right" />
                        </Button>
                        <Button
                            className={"page_button"}
                            disabled={
                                (Math.floor(this.state.topics[this.state.topicA].words.length / 20) <= this.state.word_page &&
                                    Math.floor(this.state.topics[this.state.topicB].words.length / 20) <= this.state.word_page)
                                    ? true : false}
                            onClick={this.setWordPage.bind(this, this.state.word_page + 1)}>
                            <img className="paging_button" src={require("./css/chev.png")} alt="page right" />
                        </Button>
                        <Button
                            className={"page_button"}
                            onClick={this.setWordPage.bind(this, Math.floor(this.state.topics[this.state.topicA].words.length / 20))}>
                            <img className="paging_button" src={require("./css/chev_dbl.png")} alt="page right" />
                        </Button>
                    </div> 
                    */}

                    <div id="topic_chart">
                        <TopicBarChart {...this.state}
                            setTopicA={this.setTopicA}
                            setTopicB={this.setTopicB}
                        />
                    </div>

                    <div className="joke_content">
                        <Lists {...this.state} clearSelection={this.clearSelection} />
                    </div>
                </div>

                <div id="tooltip" />
            </div>
        );
    }
}