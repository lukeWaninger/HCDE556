import React from 'react'
import BarChart from './js/BarChart'
import TopicBarChart from './js/TopicBarChart'
import ScatterPlot from './js/ScatterPlot'
import DensityPlot from './js/DensityPlot'
import Lists from './js/Lists'
import TopicSelector from './js/TopicSelector'
import { Button, Dialog } from '@blueprintjs/core'
import SketchPicker from 'react-color'
import ta from './data/topic_files/topic_23.json'
import tb from './data/topic_files/topic_49.json'
// import Filter from './js/Filter'

export default class Main extends React.Component {
    constructor(props) {
        super(props);

        const topicA = 23; // change import ta above as well
        const topicB = 49; // change import tb above as well
        let topics = {
            [topicA]: ta,
            [topicB]: tb
        };

        // for (let i = 0; i < 50; i++) {
        //     let t = require(`./data/topic_files/topic_${i + 1}.json`);
        //     topics = [...topics, t];
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
            words_per_bar: 21,
            topic_fifo: true,
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
            var el = document.getElementById("scatter_plot")
            toggle(el, s)

            var help = document.getElementById("scatter_plot_help")
            toggle(help, s)
        }

        var toggleBars = (s) => {
            var el = document.getElementById("bar_chart_container")
            toggle(el, s)  

            var help = document.getElementById("bar_chart_help")
            toggle(help, s)
        }

        var toggleTopicChart = (s) => {
            var el = document.getElementById("topic_chart")
           toggle(el, s)

           var help = document.getElementById("topic_chart_help")
           toggle(help, s)
        }

        var toggleDensity = (s) => {
            var el = document.getElementById("density_plot")
            toggle(el, s)

            var help = document.getElementById("density_plot_help")
            toggle(help, s)
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

            // hide the helps
            var help = document.getElementById("scatter_plot_help")
            toggle(help, "hide")

            help = document.getElementById("bar_chart_help")
            toggle(help, "hide")

            help = document.getElementById("topic_chart_help")
            toggle(help, "hide")

            help = document.getElementById("density_plot_help")
            toggle(help, "hide")
        }
    }

    topicFifo = (num) => {
        this.state.topic_fifo
            ? this.setTopicA(num)
            : this.setTopicB(num)
        this.state.topic_fifo ^= true;
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
                        <p>This web app visualizes the results of fitting an LDA topic model with the Reddit joke dataset - 200K jokes! The goal is to explore the topics to find out what the topics are about, and if any of them tend to score higher.</p>

                        <h6>Switching Topics</h6>
                        <p>Topics can be changed with the dropdowns in the upper left and right of the navbar. Their associated colors can also be changed by clicking on the color squares next to the dropdowns!</p>

                        <h6>Scatter Plot</h6>
                        <p>Each dot on the scatter plot represents a word, and the position represents how often it occurs in either topic. Words closer to the red line mean they occur evenly between the two topics.</p>
                        <p><strong>Cartesian Distortion</strong> is a means to scale the axes on demand allowing you to add separation in between the closely clustered words. It can be applied to the scatter plot to help compare word positions. You can enable or disable it by clicking the 'D' in the far left of the nav bar.</p>

                        <h6>Bar Plots</h6>
                        <p>Most text modeling algorithms tend to destroy the semantic context associated with their results. Unfortunately, LDA modeling is one of them. As such, we're only able to discover a predetermined number of topics but not give a topic name to bring context to the associated words. Our model performed the best when tuned to 50 topics. The bar plots are used to give you back some more of that semantic context. It's quite fun to flip through the topics and see what they're about. A longer bar indicates higher word to topic affinity and a brighter hue indicates higher word count.</p>

                        <h6>Area (Density) Plot</h6>
                        <p>The area chart shows how likely a joke written about a particular topic is going to receive more upvotes. If the area is all crammed up towards the left don't write a joke in that topic because it's unlikely to score well!</p>
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
                    <div id="scatter_plot_help" className="hidden">
                        <div className="header">
                            <h3>Relative frequency between topics</h3>
                        </div>
                        <p>
                            Here we’re able to compare how often words are used in each topic. The closer it is to a topic, the more it’s used there. Words closer to the red line indicate that they’re used more evenly between the two topics.
                        </p>
                        <p>
                            Feel free to <strong>hover</strong> over a point to see the word and the occurrences in each topic. You can <strong>zoom</strong> and <strong>pan</strong>, and if you select the ‘D’ at the top left corner you can activate <strong>cartesian distortion</strong> to shuffle the clustered words apart!
                        </p>
                    </div>

                    <div id="density_plot">
                        <DensityPlot {...this.state} />
                    </div>
                    <div id="density_plot_help" className="hidden">
                        <div className="header">
                            <h3>See how well these topics scored</h3>
                        </div>
                        <p>
                            This is a density plot showing how the two topics scored. The densities for the topics are overlapping. The more ‘weight’ a distribution has to the right indicates it has higher scores!
                        </p>                        
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

                    <div id="bar_chart_help" className="hidden">
                        <div className="header">
                            <h3>See which words make up the topic</h3>
                        </div>
                        <p>
                            Here’s the bread and butter for finding out what the topics are actually about. The bar plot on the left shows the words most associated with the topic selected on the left hand corner (and vice versa for the right). The length of the bar next to a word indicates the probability that the word is associated with the topic, but all that really means is that it’s used more in that topic than others. The lightness of the bar allows you to compare frequencies of use. A brighter hue indicates the word was used more than a darker.
                        </p>
                        <p>
                            If you <strong>hover</strong> over a word you’ll get the counts, along with a bar plot in the tooltip showing how often the word is used in other topics. If you only see one bar, that means it’s used only in that topic. Some words, like ‘get’, will be used often in many topics.
                        </p>
                        <p>
                            Finally, if you <strong>select</strong> a word, a window will pop up showing the top jokes where that word appears!
                        </p>
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
                            topicFifo={this.topicFifo}
                        />
                    </div>

                    <div className="joke_content">
                        <Lists {...this.state} clearSelection={this.clearSelection} />
                    </div>
                    <div id="topic_chart_help" className="hidden">
                        <div className="header">
                            <h3>See how all topics scored</h3>
                        </div>
                        <p>
                             Here we average the scores for each topic and put them on a bar plot. The higher the bar, the better the topic! The red line indicates the average number of upvotes for all topics, so you can see which ones do better than the rest, and which don’t make the cut. Can you tell which topic does better than all the rest?
                        </p>
                        <p>
                            You can <strong>hover</strong> over a bar to see a bit more information or <strong>click</strong> a bar to change the page topic!
                        </p>
                    </div>
                </div>

                <div id="tooltip" />
            </div>
        );
    }
}